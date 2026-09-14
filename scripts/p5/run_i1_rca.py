#!/usr/bin/env python3
"""Train frozen GAIA Ada-RCA-G and evaluate GT-anchor service ranking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.protocol import GAIA_SERVICES, load_config, sha256_file, write_json
from src.e2e.rca_model import (
    FEATURE_DIMENSION,
    GRADIENT_TOLERANCE,
    L2_LAMBDA,
    MAX_ITER,
    SOURCE_COMMIT,
    fit_conditional_logit,
    fit_root_frequency,
    predict_rankings,
    rank_candidates,
    rca_metrics,
    save_conditional_logit,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="train-oracle",
                        choices=("train-oracle", "train-v3", "smoke"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--feature-root", default="data/p5/v3/rca_features_gt")
    parser.add_argument("--case-registry", default="artifacts/p5/v3/rca/rca_case_registry_gt.csv")
    parser.add_argument("--detected-feature-root", default=None)
    parser.add_argument("--detected-case-registry", default=None)
    parser.add_argument("--model-path", default="data/p5/v3/rca_model/conditional_logit.npz")
    parser.add_argument("--artifact-root", default="artifacts/p5/v3/rca")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def _load_bundle(
    feature_root: Path,
    case_registry: Path,
    allowed_splits=None,
    required_splits=None,
):
    features_path = feature_root / "z2_features.npy"
    case_ids_path = feature_root / "case_ids.npy"
    splits_path = feature_root / "splits.npy"
    anchors_path = feature_root / "anchors_ms.npy"
    required_paths = (features_path, case_ids_path, splits_path, anchors_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("RCA features are not materialized: {}".format(missing))
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    case_ids = np.load(case_ids_path, allow_pickle=False).astype(str)
    splits = np.load(splits_path, allow_pickle=False).astype(str)
    anchors = np.load(anchors_path, allow_pickle=False).astype(np.int64)
    if features.shape != (len(case_ids), len(GAIA_SERVICES), FEATURE_DIMENSION):
        raise ValueError("RCA feature bundle does not have shape (cases, 10, 68)")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("RCA feature bundle case IDs are not unique")
    if not (len(splits) == len(case_ids) == len(anchors)):
        raise ValueError("RCA feature bundle arrays are not aligned")
    registry = pd.read_csv(case_registry)
    required = {"case_id", "service", "fault_type", "split", "start_ms"}
    if not required.issubset(registry.columns):
        raise ValueError("RCA case registry lacks required label/evaluation columns")
    if registry["case_id"].duplicated().any():
        raise ValueError("RCA case registry IDs are not unique")
    registry["case_id"] = registry["case_id"].astype(str)
    by_id = registry.set_index("case_id", drop=False)
    if set(case_ids) != set(by_id.index.astype(str)):
        raise ValueError("RCA feature bundle and case registry identity sets differ")
    ordered = by_id.loc[case_ids].reset_index(drop=True)
    if not np.array_equal(ordered["split"].astype(str).to_numpy(), splits):
        raise ValueError("RCA feature split metadata differs from case registry")
    if not np.array_equal(ordered["start_ms"].to_numpy(dtype=np.int64), anchors):
        raise ValueError("RCA feature anchors differ from GT case registry")
    roots = ordered["service"].map({name: i for i, name in enumerate(GAIA_SERVICES)})
    if roots.isna().any():
        raise ValueError("RCA case registry contains an unknown labelled service")
    names = tuple(allowed_splits or ("train", "validation", "test"))
    if not set(splits).issubset(set(names)):
        raise ValueError("RCA feature bundle contains an unsupported split")
    split_indices = {
        name: np.flatnonzero(splits == name).astype(np.int64)
        for name in names
    }
    required = tuple(required_splits or names)
    if any(len(split_indices[name]) == 0 for name in required):
        raise ValueError("every chronological split must contain RCA cases")
    return features, ordered, roots.to_numpy(dtype=np.int64), split_indices, required_paths


def _prediction_frame(
    case_rows: pd.DataFrame,
    scores: np.ndarray,
    rankings: Sequence[Sequence[str]],
    *,
    method: str,
    anchor_type: str,
) -> pd.DataFrame:
    if len(case_rows) != len(rankings) or scores.shape != (len(case_rows), len(GAIA_SERVICES)):
        raise ValueError("prediction rows, scores, and rankings are not aligned")
    rows = []
    for position, source in enumerate(case_rows.itertuples(index=False)):
        ranking = tuple(rankings[position])
        root = str(source.service)
        row = {
            "case_id": str(source.case_id),
            "split": str(source.split),
            "anchor_type": anchor_type,
            "anchor_ms": int(source.start_ms),
            "gt_start_ms": int(source.start_ms),
            "gt_service": root,
            "fault_type": str(source.fault_type),
            "method": method,
            "labelled_rank": int(ranking.index(root) + 1),
            "ranking_json": json.dumps(ranking, separators=(",", ":")),
        }
        row.update({"rank_{}".format(i + 1): service for i, service in enumerate(ranking)})
        row.update({
            "score_{}".format(service): float(scores[position, service_index])
            for service_index, service in enumerate(GAIA_SERVICES)
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_for_subset(rankings, roots, cases):
    if len(cases) == 0:
        values = {metric: None for metric in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")}
        return {
            "overall": {"case_count": 0, **values},
            "root_macro": {"by_group": {}, "group_count": 0, "macro": values.copy()},
            "fault_macro": {"by_group": {}, "group_count": 0, "macro": values.copy()},
            "case_metrics": [],
            "status": "NO_TEST_CASES",
        }
    return rca_metrics(
        rankings,
        roots,
        GAIA_SERVICES,
        cases["fault_type"].astype(str).tolist(),
    )


def _is_v3(config):
    return str(config.get("schema_version", "")).startswith("p5_v3_")


def _config_path(config, config_path=None):
    if config_path is not None:
        return Path(config_path).resolve()
    return PROJECT_ROOT / (
        "configs/e2e/gaia_p5_v3.json" if _is_v3(config)
        else "configs/e2e/gaia_p5_v1.yaml"
    )


def _prediction_frame_v3(
    case_rows: pd.DataFrame,
    scores: np.ndarray,
    rankings: Sequence[Sequence[str]],
    *,
    method: str,
    default_anchor_type: str,
) -> pd.DataFrame:
    """Serialize oracle or detected rows with explicit anchor provenance."""

    if len(case_rows) != len(rankings) or scores.shape != (len(case_rows), len(GAIA_SERVICES)):
        raise ValueError("prediction rows, scores, and rankings are not aligned")
    rows = []
    for position, source in enumerate(case_rows.itertuples(index=False)):
        ranking = tuple(rankings[position])
        root = str(source.service)
        anchor = int(source.start_ms)
        gt_start = int(getattr(source, "gt_start_ms", anchor))
        prediction_id = getattr(source, "prediction_id", None)
        if prediction_id is None or pd.isna(prediction_id):
            prediction_id = str(source.case_id)
        delay = getattr(source, "detection_delay_seconds", None)
        if delay is None or pd.isna(delay):
            delay = (anchor - gt_start) / 1000.0
        anchor_type = getattr(source, "anchor_type", default_anchor_type)
        row = {
            "prediction_id": str(prediction_id),
            "case_id": str(source.case_id),
            "split": str(source.split),
            "anchor_type": str(anchor_type),
            "anchor_ms": anchor,
            "t_hat": anchor,
            "gt_start_ms": gt_start,
            "detection_delay_seconds": float(delay),
            "gt_service": root,
            "fault_type": str(source.fault_type),
            "method": method,
            "labelled_rank": int(ranking.index(root) + 1),
            "ranking_json": json.dumps(ranking, separators=(",", ":")),
        }
        row.update({"rank_{}".format(i + 1): service for i, service in enumerate(ranking)})
        row.update({
            "score_{}".format(service): float(scores[position, service_index])
            for service_index, service in enumerate(GAIA_SERVICES)
        })
        rows.append(row)
    columns = [
        "prediction_id", "case_id", "split", "anchor_type", "anchor_ms", "t_hat",
        "gt_start_ms", "detection_delay_seconds", "gt_service", "fault_type", "method",
        "labelled_rank", "ranking_json",
    ] + ["rank_{}".format(i) for i in range(1, 11)] + [
        "score_{}".format(service) for service in GAIA_SERVICES
    ]
    return pd.DataFrame(rows, columns=columns)


def train_v3(
    config: Mapping[str, object],
    feature_root: Path,
    case_path: Path,
    model_path: Path,
    artifact_root: Path,
    detected_feature_root: Path = None,
    detected_case_path: Path = None,
    config_path: Path = None,
    formal_result: bool = True,
):
    """Fit Train-only Ada-RCA and emit Oracle, Detected, and frequency rows."""

    allowed = ("train", "test")
    features, cases, roots, indices, bundle_paths = _load_bundle(
        feature_root, case_path, allowed_splits=allowed
    )
    fit = fit_conditional_logit(
        features,
        roots,
        train_indices=indices["train"],
        l2_lambda=float(config["rca"]["l2_lambda"]),
        max_iter=int(config["rca"]["max_iter"]),
        gradient_tolerance=float(config["rca"]["gradient_tolerance"]),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_metadata = save_conditional_logit(model_path, fit)
    artifact_root.mkdir(parents=True, exist_ok=True)

    test_indices = indices["test"]
    test_cases = cases.iloc[test_indices].reset_index(drop=True)
    oracle_values = np.asarray(features[test_indices], dtype=np.float64)
    oracle_scores = fit.scores(oracle_values)
    oracle_rankings = predict_rankings(oracle_values, fit, GAIA_SERVICES)
    oracle_metrics = _metrics_for_subset(oracle_rankings, roots[test_indices], test_cases)
    oracle_frame = _prediction_frame_v3(
        test_cases, oracle_scores, oracle_rankings,
        method="Ada-RCA-G Oracle", default_anchor_type="GT injection start",
    )

    train_frequency = fit_root_frequency(
        roots, train_indices=indices["train"], candidates=GAIA_SERVICES
    )
    frequency_rankings = (
        train_frequency.predict(len(test_cases)) if len(test_cases) else tuple()
    )
    frequency_scores = (
        np.tile(
            np.asarray([train_frequency.counts[name] for name in GAIA_SERVICES], dtype=float),
            (len(test_cases), 1),
        )
        if len(test_cases) else np.empty((0, len(GAIA_SERVICES)), dtype=float)
    )
    frequency_frame = _prediction_frame_v3(
        test_cases, frequency_scores, frequency_rankings,
        method="Root Frequency (Train only)", default_anchor_type="GT injection start",
    )
    frequency_metrics = _metrics_for_subset(
        frequency_rankings, roots[test_indices], test_cases
    )
    oracle_path = artifact_root / "rca_oracle_predictions.csv"
    frequency_path = artifact_root / "root_frequency_predictions.csv"
    oracle_frame.to_csv(oracle_path, index=False, line_terminator="\n")
    frequency_frame.to_csv(frequency_path, index=False, line_terminator="\n")

    detected_path = None
    detected_metrics = None
    detected_bundle_paths = ()
    if detected_feature_root is not None or detected_case_path is not None:
        if detected_feature_root is None or detected_case_path is None:
            raise ValueError("detected feature root and case registry must be supplied together")
        detected_features, detected_cases, detected_roots, detected_indices, detected_bundle_paths = _load_bundle(
            detected_feature_root, detected_case_path, allowed_splits=allowed,
            required_splits=("train",),
        )
        detected_test = detected_indices["test"]
        detected_cases_test = detected_cases.iloc[detected_test].reset_index(drop=True)
        detected_values = np.asarray(detected_features[detected_test], dtype=np.float64)
        detected_scores = fit.scores(detected_values)
        detected_rankings = predict_rankings(detected_values, fit, GAIA_SERVICES)
        detected_metrics = _metrics_for_subset(
            detected_rankings, detected_roots[detected_test], detected_cases_test
        )
        detected_frame = _prediction_frame_v3(
            detected_cases_test, detected_scores, detected_rankings,
            method="Ada-RCA-G Detected", default_anchor_type="detected prediction_available_time",
        )
        detected_path = artifact_root / "rca_detected_predictions.csv"
        detected_frame.to_csv(detected_path, index=False, line_terminator="\n")

    config_path = _config_path(config, config_path)
    result_status = "FORMAL_FULL_DATA" if formal_result else "SMOKE_TEST_NOT_FORMAL"
    metrics = {
        "schema_version": "p5_v3_rca_metrics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result_status,
        "git_commit": git_head(),
        "config_sha256": sha256_file(config_path),
        "random_seed": int(config["random_seed"]),
        "terminology": "service is the labelled injected/fault service; no independent causal-root claim",
        "ada_rca_g": {
            "training_anchor": "GT injection start",
            "test_oracle": oracle_metrics,
            "test_detected": detected_metrics,
        },
        "root_frequency_train_only": {"test_oracle": frequency_metrics},
    }
    write_json(artifact_root / "rca_metrics.json", metrics)
    manifest = {
        "schema_version": "p5_v3_rca_train_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result_status,
        "git_commit": git_head(),
        "source_commit": SOURCE_COMMIT,
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(config_path),
        "case_registry": {"path": str(case_path.resolve()), "sha256": sha256_file(case_path)},
        "feature_bundle": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in bundle_paths
        },
        "detected_case_registry": (
            {"path": str(detected_case_path.resolve()), "sha256": sha256_file(detected_case_path)}
            if detected_case_path is not None else None
        ),
        "detected_feature_bundle": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in detected_bundle_paths
        },
        "case_counts": {name: int(len(value)) for name, value in indices.items()},
        "training": {
            "split": "train only",
            "standard_scaler_fit": "Train candidate rows only",
            "validation": "not present in V3",
            "test_use": "final evaluation only; no fitting or selection",
            "lambda": float(config["rca"]["l2_lambda"]),
            "max_iter": int(config["rca"]["max_iter"]),
            "gradient_tolerance": float(config["rca"]["gradient_tolerance"]),
            "fit": model_metadata,
        },
        "checkpoint": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "metadata_path": str(model_path.with_suffix(".json").resolve()),
            "metadata_sha256": sha256_file(model_path.with_suffix(".json")),
        },
        "root_frequency": {
            "fit_split": "train only",
            "counts": dict(train_frequency.counts),
            "ranking": list(train_frequency.ranking),
        },
        "artifacts": {
            "rca_oracle_predictions.csv": sha256_file(oracle_path),
            "root_frequency_predictions.csv": sha256_file(frequency_path),
            "rca_metrics.json": sha256_file(artifact_root / "rca_metrics.json"),
            **({"rca_detected_predictions.csv": sha256_file(detected_path)} if detected_path else {}),
        },
    }
    write_json(artifact_root / "rca_train_manifest.json", manifest)
    return manifest


def train_oracle(config: Mapping[str, object], feature_root: Path, case_path: Path,
                 model_path: Path, artifact_root: Path):
    features, cases, roots, indices, bundle_paths = _load_bundle(feature_root, case_path)
    fit = fit_conditional_logit(
        features,
        roots,
        train_indices=indices["train"],
        l2_lambda=float(config["rca"]["l2_lambda"]),
        max_iter=int(config["rca"]["max_iter"]),
        gradient_tolerance=float(config["rca"]["gradient_tolerance"]),
    )
    model_metadata = save_conditional_logit(model_path, fit)
    artifact_root.mkdir(parents=True, exist_ok=True)

    evaluated = {}
    predictions = {}
    for split in ("validation", "test"):
        selected = indices[split]
        selected_features = np.asarray(features[selected], dtype=np.float64)
        scores = fit.scores(selected_features)
        rankings = predict_rankings(selected_features, fit, GAIA_SERVICES)
        selected_cases = cases.iloc[selected].reset_index(drop=True)
        evaluated[split] = _metrics_for_subset(rankings, roots[selected], selected_cases)
        predictions[split] = (selected_cases, scores, rankings)

    train_frequency = fit_root_frequency(
        roots, train_indices=indices["train"], candidates=GAIA_SERVICES
    )
    test_cases, _, _ = predictions["test"]
    frequency_rankings = train_frequency.predict(len(test_cases))
    frequency_scores = np.tile(
        np.asarray([train_frequency.counts[name] for name in GAIA_SERVICES], dtype=float),
        (len(test_cases), 1),
    )
    oracle_frame = _prediction_frame(
        *predictions["test"], method="Ada-RCA-G", anchor_type="GT injection start"
    )
    frequency_frame = _prediction_frame(
        test_cases,
        frequency_scores,
        frequency_rankings,
        method="Root Frequency (Train only)",
        anchor_type="GT injection start",
    )
    oracle_path = artifact_root / "rca_oracle_predictions.csv"
    frequency_path = artifact_root / "root_frequency_predictions.csv"
    oracle_frame.to_csv(oracle_path, index=False)
    frequency_frame.to_csv(frequency_path, index=False)

    frequency_metrics = _metrics_for_subset(
        frequency_rankings, roots[indices["test"]], test_cases
    )
    metrics = {
        "schema_version": "p5_i1_rca_metrics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_FULL_DATA",
        "git_commit": git_head(),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "random_seed": int(config["random_seed"]),
        "terminology": "service is the labelled injected/fault service; no independent causal-root claim",
        "ada_rca_g": {
            "anchor": "GT injection start",
            "validation": evaluated["validation"],
            "test": evaluated["test"],
        },
        "root_frequency_train_only": {"test": frequency_metrics},
    }
    write_json(artifact_root / "rca_metrics.json", metrics)

    manifest = {
        "schema_version": "p5_i1_rca_train_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_FULL_DATA",
        "git_commit": git_head(),
        "source_commit": SOURCE_COMMIT,
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "case_registry": {"path": str(case_path.resolve()), "sha256": sha256_file(case_path)},
        "feature_bundle": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in bundle_paths
        },
        "case_counts": {name: int(len(value)) for name, value in indices.items()},
        "training": {
            "split": "train only",
            "standard_scaler_fit": "Train candidate rows only",
            "validation_use": "report only; no hyperparameter or checkpoint selection",
            "test_use": "final evaluation only",
            "lambda": L2_LAMBDA,
            "max_iter": MAX_ITER,
            "gradient_tolerance": GRADIENT_TOLERANCE,
            "fit": model_metadata,
        },
        "checkpoint": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "metadata_path": str(model_path.with_suffix(".json").resolve()),
            "metadata_sha256": sha256_file(model_path.with_suffix(".json")),
        },
        "root_frequency": {
            "fit_split": "train only",
            "counts": dict(train_frequency.counts),
            "ranking": list(train_frequency.ranking),
        },
        "artifacts": {
            "rca_oracle_predictions.csv": sha256_file(oracle_path),
            "root_frequency_predictions.csv": sha256_file(frequency_path),
            "rca_metrics.json": sha256_file(artifact_root / "rca_metrics.json"),
        },
    }
    write_json(artifact_root / "rca_train_manifest.json", manifest)
    return manifest


def smoke(config: Mapping[str, object], artifact_root: Path):
    rng = np.random.default_rng(int(config["random_seed"]))
    split_names = np.asarray(
        (["train"] * 12 + ["test"] * 8)
        if _is_v3(config)
        else (["train"] * 12 + ["validation"] * 4 + ["test"] * 4)
    )
    roots = np.arange(len(split_names), dtype=np.int64) % len(GAIA_SERVICES)
    features = rng.normal(0.0, 0.05, size=(len(split_names), 10, FEATURE_DIMENSION))
    for case_index, root in enumerate(roots):
        features[case_index, root, 0] += 2.0
    train = np.flatnonzero(split_names == "train")
    fit = fit_conditional_logit(features, roots, train_indices=train)
    test = np.flatnonzero(split_names == "test")
    rankings = predict_rankings(features[test], fit, GAIA_SERVICES)
    smoke_faults = ["smoke-a", "smoke-b"] * ((len(test) + 1) // 2)
    metrics = rca_metrics(
        rankings, roots[test], GAIA_SERVICES,
        smoke_faults[:len(test)],
    )
    frequency = fit_root_frequency(roots, train_indices=train, candidates=GAIA_SERVICES)
    summary = {
        "schema_version": "p5_v3_rca_training_smoke_v1" if _is_v3(config) else "p5_i1_rca_training_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "formal_result": False,
        "fixture": "synthetic only",
        "git_commit": git_head(),
        "source_commit": SOURCE_COMMIT,
        "random_seed": int(config["random_seed"]),
        "shape": list(features.shape),
        "split_counts": {
            name: int((split_names == name).sum())
            for name in (("train", "test") if _is_v3(config) else ("train", "validation", "test"))
        },
        "scaler_fit_case_indices": list(fit.train_case_indices),
        "fit": {
            "converged": bool(fit.converged),
            "gradient_norm": float(fit.gradient_norm),
            "initial_loss": float(fit.initial_loss),
            "final_loss": float(fit.final_loss),
        },
        "test_metrics": metrics,
        "root_frequency_fit_split": "train only",
        "root_frequency_ranking": list(frequency.ranking),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "rca_training_smoke_summary.json", summary)
    return summary


def main():
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    if args.action == "smoke":
        result = smoke(config, artifact_root)
    elif args.action == "train-v3":
        result = train_v3(
            config,
            (PROJECT_ROOT / args.feature_root).resolve(),
            (PROJECT_ROOT / args.case_registry).resolve(),
            (PROJECT_ROOT / args.model_path).resolve(),
            artifact_root,
            (PROJECT_ROOT / args.detected_feature_root).resolve()
            if args.detected_feature_root else None,
            (PROJECT_ROOT / args.detected_case_registry).resolve()
            if args.detected_case_registry else None,
            config_path=config_path,
        )
    else:
        if _is_v3(config):
            result = train_v3(
                config,
                (PROJECT_ROOT / args.feature_root).resolve(),
                (PROJECT_ROOT / args.case_registry).resolve(),
                (PROJECT_ROOT / args.model_path).resolve(),
                artifact_root,
                (PROJECT_ROOT / args.detected_feature_root).resolve()
                if args.detected_feature_root else None,
                (PROJECT_ROOT / args.detected_case_registry).resolve()
                if args.detected_case_registry else None,
                config_path=config_path,
            )
        else:
            result = train_oracle(
                config,
                (PROJECT_ROOT / args.feature_root).resolve(),
                (PROJECT_ROOT / args.case_registry).resolve(),
                (PROJECT_ROOT / args.model_path).resolve(),
                artifact_root,
            )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
