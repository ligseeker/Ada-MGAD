#!/usr/bin/env python3
"""Prepare, smoke-test, train, and infer the P5-I1 Ada-MGAD-G stage."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.ad_data import (
    PaddedSequentialSampler,
    build_semisupervised_mask,
    load_timestamped_datasets,
    save_split_arrays,
)
from src.e2e.ad_preprocess import build_ad_data
from src.e2e.protocol import GAIA_SERVICES, load_config, sha256_file, write_json
from src.model import MyModel
from util.train import MY
from util.util import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preprocess", "smoke", "train", "infer", "all"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--data-root", default="data/p5/i1/ad")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument("--checkpoint-dir", default="data/p5/i1/checkpoint")
    parser.add_argument(
        "--raw-root", default=None,
        help="Optional byte-layout-verified execution mirror of the canonical GAIA raw root.",
    )
    parser.add_argument("--chunk-rows", default=500000, type=int)
    parser.add_argument("--gpu", default=True, type=lambda value: value.lower() == "true")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def model_args(config, manifest, checkpoint_dir: Path, gpu: bool, overrides=None):
    args = dict(config["ad_model"])
    args.update({
        "random_seed": int(config["random_seed"]),
        "gpu": bool(gpu),
        "epochs": int(config["ad"]["epochs"]),
        "patience": float(config["ad"]["patience"]),
        "window": int(config["ad"]["window_bins"]),
        "step": int(config["ad"]["window_step"]),
        "num_nodes": len(GAIA_SERVICES),
        "raw_node": int(manifest["dimensions"]["raw_node"]),
        "log_len": int(manifest["dimensions"]["log_len"]),
        "raw_edge": int(manifest["dimensions"]["raw_edge"]),
        "result_dir": str(checkpoint_dir.resolve()),
        "model_path": str(checkpoint_dir.resolve()),
        "main_model": "Ada-MGAD-G",
        "evaluate": False,
    })
    if overrides:
        args.update(overrides)
    return args


def build_loaders(datasets, args):
    batch_size = int(args["batch_size"])
    common = {
        "batch_size": batch_size,
        "num_workers": int(args["num_workers"]),
        "pin_memory": bool(args["pin_memory"] and args["gpu"]),
    }
    if int(args["num_workers"]) > 0:
        common["persistent_workers"] = bool(args["persistent_workers"])
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args["random_seed"]))
    train = DataLoader(
        datasets["train"], shuffle=True, drop_last=True,
        generator=train_generator, **common
    )
    validation = DataLoader(
        datasets["validation"],
        sampler=PaddedSequentialSampler(datasets["validation"], batch_size),
        drop_last=False, **common
    )
    test = DataLoader(
        datasets["test"],
        sampler=PaddedSequentialSampler(datasets["test"], batch_size),
        drop_last=False, **common
    )
    return train, validation, test


def _deduplicate(indices, scores, labels):
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    scores = scores[order]
    labels = labels[order]
    keep = np.ones(len(indices), dtype=bool)
    keep[1:] = indices[1:] != indices[:-1]
    return indices[keep], scores[keep], labels[keep]


def timestamped_predict(system: MY, loader, dataset):
    system.model.eval()
    all_indices = []
    all_scores = []
    all_labels = []
    reconstruction = []
    with torch.no_grad():
        for batch in loader:
            batch = system.input2device(batch, system.use_gpu)
            all_indices.append(batch["sample_index"].reshape(-1).long().cpu().numpy())
            if system.score_fusion_alpha < 1.0:
                scores, _, rec = system.model(batch, evaluate=True, return_eval_aux=True)
                reconstruction.append(rec.reshape(-1).cpu())
            else:
                scores, _ = system.model(batch, evaluate=True)
            all_scores.append(scores.cpu())
            all_labels.append(batch["groundtruth_real"].cpu())
    score_tensor = torch.concat(all_scores, dim=0)
    if reconstruction:
        score_tensor = system._fuse_predict_with_reconstruction(
            score_tensor, torch.concat(reconstruction, dim=0)
        )
    indices = np.concatenate(all_indices).astype(np.int64)
    scores = score_tensor.numpy()
    labels = torch.concat(all_labels, dim=0).numpy()
    indices, scores, labels = _deduplicate(indices, scores, labels)
    if len(indices) != len(dataset) or not np.array_equal(indices, np.arange(len(dataset))):
        raise ValueError("timestamped prediction did not cover every window exactly once")
    return indices, scores, labels


def node_metrics(scores: np.ndarray, labels: np.ndarray):
    anomaly = scores[..., 1].reshape(-1)
    actual = np.argmax(labels, axis=-1).reshape(-1)
    predicted = (anomaly >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(actual, anomaly))
    except ValueError:
        auc = None
    try:
        ap = float(average_precision_score(actual, anomaly))
    except ValueError:
        ap = None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "average_precision": ap,
        "node_rows": int(len(actual)),
        "positive_labels": int(actual.sum()),
        "positive_predictions": int(predicted.sum()),
    }


def write_timestamped_predictions(path: Path, dataset, indices, scores, labels):
    rows = []
    for position, sample_index in enumerate(indices):
        metadata = dataset.metadata(int(sample_index))
        for service_index, service in enumerate(GAIA_SERVICES):
            probability = float(scores[position, service_index, 1])
            rows.append({
                "split": metadata.split,
                "sample_index": int(sample_index),
                "window_start_time": metadata.window_start_time,
                "window_end_time": metadata.window_end_time,
                "prediction_timestamp": metadata.prediction_timestamp,
                "service": service,
                "service_registry_index": service_index,
                "anomaly_score": probability,
                "node_label": int(np.argmax(labels[position, service_index])),
                "binary_prediction": int(probability >= 0.5),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_manifest(artifact_root: Path):
    return json.loads((artifact_root / "ad_data_manifest.json").read_text())


def evaluate_checkpoint(config, data_root, artifact_root, checkpoint_dir, gpu):
    manifest = load_manifest(artifact_root)
    args = model_args(config, manifest, checkpoint_dir, gpu)
    seed_everything(int(args["random_seed"]))
    datasets = load_timestamped_datasets(
        data_root, int(args["window"]), int(config["ad"]["grid_seconds"])
    )
    loaders = build_loaders(datasets, args)
    graph = np.load(data_root / "graph.npy")
    system = MY(MyModel(graph, **args), **args)
    system.load_model(str(checkpoint_dir), name="f1")
    outputs = {}
    for split, loader in (("validation", loaders[1]), ("test", loaders[2])):
        indices, scores, labels = timestamped_predict(system, loader, datasets[split])
        path = artifact_root / ("ad_{}_predictions.csv".format(split))
        write_timestamped_predictions(path, datasets[split], indices, scores, labels)
        outputs[split] = {
            "indices": indices,
            "scores": scores,
            "labels": labels,
            "path": path,
            "metrics": node_metrics(scores, labels),
        }
    return args, datasets, outputs


def train_and_infer(config, data_root: Path, artifact_root: Path, checkpoint_dir: Path, gpu: bool):
    manifest = load_manifest(artifact_root)
    args = model_args(config, manifest, checkpoint_dir, gpu)
    seed_everything(int(args["random_seed"]))
    datasets = load_timestamped_datasets(
        data_root, int(args["window"]), int(config["ad"]["grid_seconds"])
    )
    train_loader, validation_loader, _ = build_loaders(datasets, args)
    graph = np.load(data_root / "graph.npy")
    system = MY(MyModel(graph, **args), **args)
    fit_summary = system.fit(train_loader=train_loader, val_loader=validation_loader)
    checkpoint = checkpoint_dir / "Ada-MGAD-G_f1_stage.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    eval_args, datasets, outputs = evaluate_checkpoint(
        config, data_root, artifact_root, checkpoint_dir, gpu
    )
    summary = {
        "schema_version": "p5_i1_ad_training_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "random_seed": int(args["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "data_manifest_sha256": sha256_file(artifact_root / "ad_data_manifest.json"),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "committed_to_git": False,
        },
        "checkpoint_selection": "validation node F1 only; Test not evaluated during fit",
        "fit": fit_summary,
        "split_window_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "validation_node_metrics": outputs["validation"]["metrics"],
        "test_node_metrics": outputs["test"]["metrics"],
        "prediction_artifacts": {
            split: {
                "path": str(output["path"].resolve()),
                "sha256": sha256_file(output["path"]),
            }
            for split, output in outputs.items()
        },
        "model_args": eval_args,
    }
    write_json(artifact_root / "ad_training_summary.json", summary)
    return summary


def smoke(config, data_root: Path, artifact_root: Path, gpu: bool):
    smoke_root = data_root.parent / "ad_smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    rng = np.random.RandomState(int(config["random_seed"]))
    for split_index, split in enumerate(("train", "validation", "test")):
        count = 48
        timestamps = np.arange(count, dtype=np.int64) * 30000 + split_index * 10_000_000
        metric = rng.normal(size=(count, 10, 4)).astype(np.float32)
        logs = rng.uniform(size=(count, 10, 6)).astype(np.float32)
        trace = rng.uniform(size=(count, 10, 10, 4)).astype(np.float32)
        labels = np.zeros((count, 10), dtype=np.int8)
        labels[15:18, split_index] = 1
        masks = build_semisupervised_mask(labels, 0.5, 10)
        save_split_arrays(smoke_root, split, {
            "timestamps": timestamps,
            "metric": metric,
            "log": logs,
            "trace": trace,
            "labels": labels,
            "label_mask": masks,
        })
    graph = np.zeros((10, 10), dtype=np.float32)
    for index in range(10):
        graph[index, (index + 1) % 10] = 1.0
        graph[(index + 1) % 10, index] = 1.0
    np.save(smoke_root / "graph.npy", graph, allow_pickle=False)
    manifest = {"dimensions": {"raw_node": 4, "log_len": 6, "raw_edge": 4}}
    datasets = load_timestamped_datasets(smoke_root, 10, 30)
    checkpoint_dir = smoke_root / "checkpoint"
    args = model_args(config, manifest, checkpoint_dir, gpu, {
        "epochs": 3,
        "patience": 1,
        "batch_size": 32,
        "num_workers": 0,
        "train_eval_interval": 0,
    })
    seed_everything(int(args["random_seed"]))
    loaders = build_loaders(datasets, args)
    system = MY(MyModel(graph, **args), **args)
    fit = system.fit(train_loader=loaders[0], val_loader=loaders[1])
    system.load_model(str(checkpoint_dir), name="f1")
    indices, scores, labels = timestamped_predict(system, loaders[2], datasets["test"])
    summary = {
        "schema_version": "p5_i1_ad_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "random_seed": int(config["random_seed"]),
        "status": "PASS",
        "formal_result": False,
        "fixture": "synthetic only",
        "gpu": bool(gpu and torch.cuda.is_available()),
        "windows_per_split": {name: len(dataset) for name, dataset in datasets.items()},
        "prediction_windows": len(indices),
        "prediction_shape": list(scores.shape),
        "finite_scores": bool(np.isfinite(scores).all()),
        "fit": fit,
        "node_metrics": node_metrics(scores, labels),
    }
    if not summary["finite_scores"]:
        raise ValueError("Ada-MGAD smoke produced non-finite scores")
    write_json(artifact_root / "ad_smoke_summary.json", summary)
    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config((PROJECT_ROOT / args.config).resolve())
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    checkpoint_dir = (PROJECT_ROOT / args.checkpoint_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if args.action in ("preprocess", "all"):
        build_ad_data(
            config, PROJECT_ROOT, data_root, artifact_root, args.chunk_rows,
            Path(args.raw_root).resolve() if args.raw_root else None,
        )
    if args.action == "smoke":
        result = smoke(config, data_root, artifact_root, args.gpu)
    elif args.action in ("train", "all"):
        result = train_and_infer(config, data_root, artifact_root, checkpoint_dir, args.gpu)
    elif args.action == "infer":
        eval_args, datasets, outputs = evaluate_checkpoint(
            config, data_root, artifact_root, checkpoint_dir, args.gpu
        )
        result = {
            "checkpoint": str((checkpoint_dir / "Ada-MGAD-G_f1_stage.ckpt").resolve()),
            "validation_node_metrics": outputs["validation"]["metrics"],
            "test_node_metrics": outputs["test"]["metrics"],
        }
    else:
        result = {"status": "PREPROCESS_COMPLETE"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
