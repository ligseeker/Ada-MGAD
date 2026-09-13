#!/usr/bin/env python3
"""Run the V3 Ada-MGAD detector stage on Train/Test arrays."""

from __future__ import annotations

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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
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
from src.e2e.calibration import (
    fit_reconstruction_calibration,
    save_reconstruction_calibration,
    load_reconstruction_calibration,
)
from src.e2e.protocol import GAIA_SERVICES, load_config, preprocessing_runtime, sha256_file, write_json
from src.model import MyModel
from util.train import MY
from util.util import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preprocess", "smoke", "train", "infer", "all"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--data-root", default="data/p5/v3/ad")
    parser.add_argument("--artifact-root", default="artifacts/p5/v3/ad")
    parser.add_argument("--checkpoint-dir", default="data/p5/v3/checkpoint")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--chunk-rows", default=None, type=int)
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default=None)
    parser.add_argument("--gpu", default=False, type=lambda value: value.lower() == "true")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()


def model_args(config, manifest, checkpoint_dir: Path, gpu: bool, overrides=None):
    args = dict(config["ad_model"])
    args.update({
        "random_seed": int(config["random_seed"]), "gpu": bool(gpu),
        "epochs": int(config["ad"]["epochs"]), "patience": int(config["ad"]["patience"]),
        "early_stopping_metric": str(config["ad"]["early_stopping_metric"]),
        "window": int(config["ad"]["window_bins"]), "step": int(config["ad"]["window_step"]),
        "num_nodes": len(GAIA_SERVICES), "raw_node": int(manifest["dimensions"]["raw_node"]),
        "log_len": int(manifest["dimensions"]["log_len"]),
        "raw_edge": int(manifest["dimensions"]["raw_edge"]),
        "result_dir": str(checkpoint_dir.resolve()), "model_path": str(checkpoint_dir.resolve()),
        "main_model": "Ada-MGAD-G", "evaluate": False,
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
    train = DataLoader(
        # The upstream model constructs a fixed batch-size graph index.  Pad
        # only within Train, so every real window participates in the epoch and
        # no partial batch can address a non-existent graph node.
        datasets["train"], sampler=PaddedSequentialSampler(datasets["train"], batch_size),
        drop_last=False, **common
    )
    train_eval = DataLoader(
        datasets["train"], sampler=PaddedSequentialSampler(datasets["train"], batch_size),
        drop_last=False, **common
    )
    test = DataLoader(
        datasets["test"], sampler=PaddedSequentialSampler(datasets["test"], batch_size),
        drop_last=False, **common
    )
    return {"train": train, "train_eval": train_eval, "test": test}


def _deduplicate(indices, *values):
    order = np.argsort(indices, kind="stable")
    sorted_indices = np.asarray(indices)[order]
    keep = np.ones(len(sorted_indices), dtype=bool)
    if len(keep) > 1:
        keep[1:] = sorted_indices[1:] != sorted_indices[:-1]
    return (sorted_indices[keep],) + tuple(np.asarray(value)[order][keep] for value in values)


def timestamped_predict(system: MY, loader, dataset, calibration=None):
    """Collect one prediction per window and apply the supplied frozen calibration."""

    system.model.eval()
    all_indices, all_scores, all_labels, all_reconstruction = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = system.input2device(batch, system.use_gpu)
            scores, _, reconstruction = system.model(batch, evaluate=True, return_eval_aux=True)
            all_indices.append(batch["sample_index"].reshape(-1).long().cpu().numpy())
            all_scores.append(scores.cpu())
            all_labels.append(batch["groundtruth_real"].cpu().numpy())
            all_reconstruction.append(reconstruction.cpu().numpy())
    if not all_indices:
        raise ValueError("prediction loader yielded no rows")
    indices = np.concatenate(all_indices).astype(np.int64)
    scores = torch.concat(all_scores, dim=0).numpy()
    labels = np.concatenate(all_labels, axis=0)
    reconstruction = np.concatenate(all_reconstruction, axis=0)
    indices, scores, labels, reconstruction = _deduplicate(indices, scores, labels, reconstruction)
    if len(indices) != len(dataset) or not np.array_equal(indices, np.arange(len(dataset))):
        raise ValueError("timestamped prediction did not cover every window exactly once")
    raw_scores = reconstruction.reshape(-1)
    if calibration is not None:
        score_tensor = torch.as_tensor(scores, dtype=torch.float32)
        reconstruction_tensor = torch.as_tensor(reconstruction, dtype=torch.float32)
        scores = system._fuse_predict_with_reconstruction(
            score_tensor, reconstruction_tensor, calibration=calibration
        ).numpy()
    return indices, scores, labels, raw_scores


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
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auc": auc, "average_precision": ap, "node_rows": int(len(actual)),
        "positive_labels": int(actual.sum()), "positive_predictions": int(predicted.sum()),
    }


def write_timestamped_predictions(path: Path, dataset, indices, scores, labels):
    rows = []
    for position, sample_index in enumerate(indices):
        metadata = dataset.metadata(int(sample_index))
        for service_index, service in enumerate(GAIA_SERVICES):
            probability = float(scores[position, service_index, 1])
            rows.append({
                "split": metadata.split, "sample_index": int(sample_index),
                "window_start_time": metadata.window_start_time, "window_end_time": metadata.window_end_time,
                "target_bin_start": metadata.target_bin_start, "target_bin_end": metadata.target_bin_end,
                "prediction_available_time": metadata.prediction_available_time,
                "prediction_timestamp": metadata.prediction_available_time,
                "service": service, "service_registry_index": service_index,
                "anomaly_score": probability,
                "node_label": int(np.argmax(labels[position, service_index])),
                "binary_prediction": int(probability >= 0.5),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def load_manifest(artifact_root: Path, config_path: Path = None):
    manifest_path = artifact_root / "ad_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("schema_version", "")).startswith("p5_v3_"):
        raise ValueError("Ada-MGAD loader requires a V3 data manifest")
    expected_config = Path(config_path or (PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json"))
    if str(manifest.get("config_sha256")) != sha256_file(expected_config):
        raise ValueError("Ada-MGAD data manifest config SHA differs from execution config")
    for name, record in manifest.get("schema_artifacts", {}).items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise ValueError("Ada-MGAD schema artifact checksum mismatch: {}".format(name))
    graph_record = manifest.get("graph", {})
    graph_path = Path(graph_record.get("path", ""))
    if not graph_path.is_file() or sha256_file(graph_path) != str(graph_record.get("sha256")):
        raise ValueError("Ada-MGAD graph artifact checksum mismatch")
    if set(manifest.get("split_counts", {})) != {"train", "test"}:
        raise ValueError("Ada-MGAD data manifest must contain exactly Train/Test splits")
    return manifest


def _load_datasets_and_system(
    config, data_root, artifact_root, checkpoint_dir, gpu, config_path=None
):
    config_path = Path(config_path or (PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json")).resolve()
    manifest = load_manifest(artifact_root, config_path)
    args = model_args(config, manifest, checkpoint_dir, gpu)
    seed_everything(int(args["random_seed"]))
    datasets = load_timestamped_datasets(
        data_root, int(args["window"]), int(config["ad"]["grid_seconds"])
    )
    loaders = build_loaders(datasets, args)
    graph = np.load(data_root / "graph.npy", allow_pickle=False)
    system = MY(MyModel(graph, **args), **args)
    return manifest, args, datasets, loaders, system


def _predict_splits(system, datasets, loaders, artifact_root, calibration):
    outputs = {}
    for split in ("train", "test"):
        indices, scores, labels, raw_scores = timestamped_predict(
            system, loaders["train_eval"] if split == "train" else loaders[split],
            datasets[split], calibration=calibration
        )
        path = artifact_root / "ad_{}_predictions.csv".format(split)
        write_timestamped_predictions(path, datasets[split], indices, scores, labels)
        outputs[split] = {
            "indices": indices, "scores": scores, "labels": labels,
            "raw_reconstruction_scores": raw_scores, "path": path,
            "metrics": node_metrics(scores, labels),
        }
    return outputs


def train_and_infer(
    config, data_root: Path, artifact_root: Path, checkpoint_dir: Path,
    gpu: bool, config_path: Path = None,
):
    manifest, args, datasets, loaders, system = _load_datasets_and_system(
        config, data_root, artifact_root, checkpoint_dir, gpu, config_path
    )
    fit_summary = system.fit(train_loader=loaders["train"], train_eval_loader=loaders["train_eval"])
    primary = checkpoint_dir / "best_train_loss.pt"
    auxiliary = checkpoint_dir / "best_train_f1.pt"
    last = checkpoint_dir / "last.pt"
    for path in (primary, auxiliary, last):
        if not path.is_file():
            raise FileNotFoundError(path)
    system.load_model(str(checkpoint_dir), name="best_train_loss")
    _, _, _, train_raw = timestamped_predict(system, loaders["train_eval"], datasets["train"], calibration=None)
    calibration = fit_reconstruction_calibration(train_raw)
    calibration_path = artifact_root / "reconstruction_calibration.json"
    save_reconstruction_calibration(calibration_path, calibration)
    outputs = _predict_splits(system, datasets, loaders, artifact_root, calibration)
    config_path = Path(
        config_path or (PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json")
    ).resolve()
    summary = {
        "schema_version": "p5_v3_ad_training_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_head(),
        "random_seed": int(args["random_seed"]), "config_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(artifact_root / "ad_data_manifest.json"),
        "schema_artifacts": manifest.get("schema_artifacts", {}),
        "graph_artifact": manifest.get("graph", {}),
        "checkpoint_policy": {
            "primary": "best_train_loss.pt", "auxiliary": "best_train_f1.pt", "last": "last.pt",
            "primary_selection": "minimum complete Train epoch average train_total_loss",
            "test_used_for_fit_or_selection": False, "validation_split": False,
        },
        "checkpoints": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (primary, auxiliary, last)
        },
        "reconstruction_calibration": {
            "path": str(calibration_path.resolve()), "sha256": sha256_file(calibration_path),
            "fit_split": "train", "test_used_for_fit": False,
        },
        "fit": fit_summary,
        "split_window_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "train_node_metrics": outputs["train"]["metrics"], "test_node_metrics": outputs["test"]["metrics"],
        "prediction_artifacts": {
            split: {"path": str(value["path"].resolve()), "sha256": sha256_file(value["path"])}
            for split, value in outputs.items()
        },
        "model_args": args,
    }
    write_json(artifact_root / "ad_training_summary.json", summary)
    return summary


def evaluate_checkpoint(config, data_root, artifact_root, checkpoint_dir, gpu, config_path=None):
    manifest, args, datasets, loaders, system = _load_datasets_and_system(
        config, data_root, artifact_root, checkpoint_dir, gpu, config_path
    )
    primary = checkpoint_dir / "best_train_loss.pt"
    summary_path = artifact_root / "ad_training_summary.json"
    if not primary.is_file() or not summary_path.is_file():
        raise FileNotFoundError("V3 primary checkpoint or training summary is missing")
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = Path(
        config_path or (PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json")
    ).resolve()
    if training_summary.get("config_sha256") != sha256_file(config_path):
        raise ValueError("training summary config SHA differs from execution config")
    expected_checkpoint_sha = training_summary.get("checkpoints", {}).get("best_train_loss.pt", {}).get("sha256")
    if expected_checkpoint_sha != sha256_file(primary):
        raise ValueError("primary checkpoint checksum differs from training provenance")
    system.load_model(str(checkpoint_dir), name="best_train_loss")
    calibration = load_reconstruction_calibration(artifact_root / "reconstruction_calibration.json")
    outputs = _predict_splits(system, datasets, loaders, artifact_root, calibration)
    return args, datasets, outputs


def smoke(config, data_root: Path, artifact_root: Path, gpu: bool):
    """Run a tiny synthetic Train/Test detector path; never a formal result."""

    smoke_root = data_root.parent / "ad_smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    rng = np.random.RandomState(int(config["random_seed"]))
    for split_index, split in enumerate(("train", "test")):
        count = 48
        timestamps = np.arange(count, dtype=np.int64) * 30000 + split_index * 10_000_000
        metric = rng.normal(size=(count, 10, 4)).astype(np.float32)
        logs = rng.uniform(size=(count, 10, 6)).astype(np.float32)
        trace = rng.uniform(size=(count, 10, 10, 4)).astype(np.float32)
        labels = np.zeros((count, 10), dtype=np.int8)
        labels[15:18, split_index] = 1
        save_split_arrays(smoke_root, split, {
            "timestamps": timestamps, "metric": metric, "log": logs, "trace": trace,
            "labels": labels, "label_mask": build_semisupervised_mask(labels, 0.5, 10),
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
        "epochs": 3, "patience": 1, "batch_size": 32, "num_workers": 0,
        "train_eval_interval": 1,
    })
    seed_everything(int(args["random_seed"]))
    loaders = build_loaders(datasets, args)
    system = MY(MyModel(graph, **args), **args)
    fit = system.fit(train_loader=loaders["train"], train_eval_loader=loaders["train_eval"])
    system.load_model(str(checkpoint_dir), name="best_train_loss")
    _, _, _, train_raw = timestamped_predict(system, loaders["train_eval"], datasets["train"])
    calibration = fit_reconstruction_calibration(train_raw)
    save_reconstruction_calibration(smoke_root / "reconstruction_calibration.json", calibration)
    outputs = _predict_splits(system, datasets, loaders, smoke_root, calibration)
    summary = {
        "schema_version": "p5_v3_ad_smoke_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(), "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json"),
        "random_seed": int(config["random_seed"]), "status": "PASS", "formal_result": False,
        "fixture": "synthetic Train/Test only", "gpu": bool(gpu and torch.cuda.is_available()),
        "windows_per_split": {name: len(dataset) for name, dataset in datasets.items()},
        "prediction_windows": {name: len(value["indices"]) for name, value in outputs.items()},
        "finite_scores": bool(all(np.isfinite(value["scores"]).all() for value in outputs.values())),
        "checkpoints": sorted(path.name for path in checkpoint_dir.glob("*.pt")),
        "fit": fit,
        "node_metrics": {name: value["metrics"] for name, value in outputs.items()},
    }
    if not summary["finite_scores"] or sorted(summary["checkpoints"]) != ["best_train_f1.pt", "best_train_loss.pt", "last.pt"]:
        raise ValueError("Ada-MGAD smoke did not produce finite scores and all V3 checkpoints")
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
    runtime = preprocessing_runtime(
        config, "ad", workers=args.workers, chunk_rows=args.chunk_rows,
        start_method=args.start_method,
    )
    if args.action in ("preprocess", "all"):
        build_ad_data(
            config, PROJECT_ROOT, data_root, artifact_root, runtime["chunk_rows"],
            Path(args.raw_root).resolve() if args.raw_root else None,
            workers=runtime["workers"], start_method=runtime["start_method"],
            # A single --workers override is intentionally global.  Without
            # it, preserve each modality's frozen V3 worker budget (notably
            # trace=24 versus metric/log=8).
            metric_workers=(args.workers if args.workers is not None else int(config["preprocessing"]["metric_workers"])),
            log_workers=(args.workers if args.workers is not None else int(config["preprocessing"]["log_workers"])),
            trace_workers=(args.workers if args.workers is not None else int(config["preprocessing"]["trace_workers"])),
            config_path=(PROJECT_ROOT / args.config).resolve(),
        )
    if args.action == "smoke":
        result = smoke(config, data_root, artifact_root, args.gpu)
    elif args.action in ("train", "all"):
        result = train_and_infer(
            config, data_root, artifact_root, checkpoint_dir, args.gpu,
            (PROJECT_ROOT / args.config).resolve(),
        )
    elif args.action == "infer":
        eval_args, datasets, outputs = evaluate_checkpoint(
            config, data_root, artifact_root, checkpoint_dir, args.gpu,
            (PROJECT_ROOT / args.config).resolve(),
        )
        result = {
            "checkpoint": str((checkpoint_dir / "best_train_loss.pt").resolve()),
            "train_node_metrics": outputs["train"]["metrics"],
            "test_node_metrics": outputs["test"]["metrics"],
        }
    elif args.action == "preprocess":
        result = {"status": "PREPROCESS_COMPLETE", "formal_result": False}
    else:
        result = {"status": "NOOP"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
