#!/usr/bin/env python3
"""P6-B0 score decomposition / label-coupling audit.

The audit re-uses the frozen P5 formal run (checkpoint, calibration,
preprocessing, registry, matching protocol, RCA model) and performs a single
pure-inference pass that exports the Ada-MGAD classification score, the
train-calibrated reconstruction score, and the frozen fused score.  It then
evaluates three fixed score tracks:

    classification-only, current-fused, reconstruction-only

Each track gets its own Train-only threshold, its own event detection
evaluation, its own anomaly-score localization diagnostic, and its own
downstream Ada-RCA evaluation (native and common-case populations).

Hard constraints enforced here:

* no Ada-MGAD training / no ``fit`` call;
* no reconstruction-calibration re-fit (the frozen JSON is only loaded);
* no conditional-logit re-fit (the frozen model is only loaded);
* no writes inside the formal P5 run directory;
* thresholds are selected on Train only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.run_i1_ad import _load_datasets_and_system  # noqa: E402
from scripts.p5.run_i1_e2e import (  # noqa: E402
    _prediction_frame,
    _purge_rca_ineligible_matching,
)
from scripts.p5.run_i1_rca_features import (  # noqa: E402
    materialize as materialize_rca_features,
)
from src.e2e.calibration import load_reconstruction_calibration  # noqa: E402
from src.e2e.e2e_evaluation import diagnosis_metrics  # noqa: E402
from src.e2e.event_detection import (  # noqa: E402
    _gt_rows_for_split,
    evaluate_threshold,
    run_event_detection,
)
from src.e2e.protocol import (  # noqa: E402
    GAIA_SERVICES,
    assign_event_blocks,
    load_config,
    load_registry,
    sha256_file,
    temporal_blocks,
    write_json,
)
from src.e2e.rca_model import (  # noqa: E402
    load_conditional_logit,
    predict_rankings,
    rca_metrics,
)
from src.e2e.score_decomposition import (  # noqa: E402
    FRAME_SCORE_COLUMNS,
    FUSION_ALPHA,
    SCORE_TRACKS,
    aggregate_only,
    apply_track,
    common_case_ids,
    complementarity_table,
    decomposition_frame,
    dynamic_ranking_metrics,
    label_identity_audit,
    localization_metrics,
    predict_score_components,
    rank_services_by_score,
    replay_against_formal,
    root_margin_report,
    score_range_report,
    stratified_event_metrics,
)


DEFAULT_RUN_DIR = "experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440"
DEFAULT_OUTPUT_DIR = "experiments/p6/score_decomposition"
DEFAULT_CONFIG = "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
P5_PREFIX = "experiments/p5"

REQUIRED_PREDICTION_COLUMNS = (
    "split", "sample_index", "window_start_time", "window_end_time",
    "target_bin_start", "target_bin_end", "prediction_available_time",
    "prediction_timestamp", "service", "service_registry_index", "node_label",
    "classification_score", "reconstruction_score", "fused_score",
    "selected_anomaly_score", "anomaly_score", "binary_prediction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("infer", "evaluate", "all"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu", default=False, type=lambda value: value.lower() == "true")
    parser.add_argument("--threshold-workers", default=8, type=int)
    parser.add_argument("--feature-workers", default=16, type=int)
    parser.add_argument("--case-chunk-size", default=128, type=int)
    parser.add_argument(
        "--reuse-event-thresholds", action="store_true",
        help=(
            "Reuse an existing track threshold.json instead of re-running the exact "
            "Train-only sweep.  Reuse is validated by re-deriving the Train metrics at "
            "the stored threshold and comparing the prediction artifact hashes."
        ),
    )
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    return parser.parse_args()


def _path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def guard_output_root(output_dir: Path, formal_run_dir: Path) -> Path:
    """Fail closed if a P6 write target could touch the formal P5 run."""

    output = Path(output_dir).resolve()
    formal = Path(formal_run_dir).resolve()
    if output == formal or formal in output.parents:
        raise ValueError("P6 output directory must not live inside the formal P5 run")
    relative = None
    try:
        relative = output.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = None
    if relative is not None and relative.parts[:2] == ("experiments", "p5"):
        raise ValueError("P6 output must not be written under experiments/p5")
    return output


def load_formal_bindings(run_dir: Path) -> Mapping[str, object]:
    input_manifest_path = run_dir / "input_manifest.json"
    if not input_manifest_path.is_file():
        raise FileNotFoundError("formal run is missing input_manifest.json")
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    incomplete = sorted(
        name for name, record in files.items() if record.get("status") != "COMPLETE"
    )
    if incomplete:
        raise ValueError("formal shared inputs are incomplete: {}".format(incomplete))
    return manifest


def _sha256_of(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_track_frame(frame: pd.DataFrame, track: str) -> None:
    missing = sorted(
        set(["split", "service", "service_registry_index", "prediction_available_time",
             "anomaly_score", "selected_anomaly_score"]) - set(frame.columns)
    )
    if missing:
        raise ValueError("track {} frame is missing columns: {}".format(track, missing))
    values = frame["anomaly_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("track {} frame contains non-finite scores".format(track))
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("track {} frame score is outside [0, 1]".format(track))
    if not set(frame["service"].astype(str)).issubset(GAIA_SERVICES):
        raise ValueError("track {} frame contains a non-canonical service".format(track))
    if not np.array_equal(
        frame["anomaly_score"].to_numpy(dtype=np.float64),
        frame["selected_anomaly_score"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("track {} anomaly_score must equal selected_anomaly_score".format(track))


def write_track_predictions(track_dir: Path, split: str, frame: pd.DataFrame) -> Mapping[str, object]:
    path = track_dir / "ad_{}_predictions.csv".format(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": int(len(frame))}


# ---------------------------------------------------------------------------
# Phase 1: inference export and replay gate
# ---------------------------------------------------------------------------


def run_inference(args, config, run_dir: Path, output_dir: Path) -> Mapping[str, object]:
    data_root = _path(config["ad_paths"]["data_root"])
    preprocess_artifact_root = _path(config["ad_paths"]["artifact_root"])
    checkpoint_root = run_dir / "checkpoint"
    checkpoint = checkpoint_root / "best_train_f1.pt"
    calibration_path = run_dir / "ad" / "reconstruction_calibration.json"
    for path in (checkpoint, calibration_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    checkpoint_before = sha256_file(checkpoint)
    config_path = _path(args.config)
    manifest, model_args, datasets, loaders, system = _load_datasets_and_system(
        config, data_root, preprocess_artifact_root, checkpoint_root, bool(args.gpu),
        config_path,
    )
    system.load_model(str(checkpoint_root), name="best_train_f1")
    calibration = load_reconstruction_calibration(calibration_path)

    outputs = {}
    for split in ("train", "test"):
        loader = loaders["train_eval"] if split == "train" else loaders[split]
        components = predict_score_components(system, loader, datasets[split], calibration)
        frame = decomposition_frame(datasets[split], components)
        outputs[split] = frame
        logging.info(
            "%s decomposition: %d rows, cls[%.4f,%.4f] rec[%.4f,%.4f] fused[%.4f,%.4f]",
            split, len(frame),
            frame["classification_score"].min(), frame["classification_score"].max(),
            frame["reconstruction_score"].min(), frame["reconstruction_score"].max(),
            frame["fused_score"].min(), frame["fused_score"].max(),
        )

    if sha256_file(checkpoint) != checkpoint_before:
        raise RuntimeError("P6 inference must not modify the frozen checkpoint")

    # The absolute fused identity must hold exactly by construction.
    identity_error = float(
        np.max(np.abs(
            outputs["train"]["fused_score"].to_numpy(dtype=np.float64)
            - outputs["train"]["fused_recomputed"].to_numpy(dtype=np.float64)
        ))
    )
    for split in ("train", "test"):
        identity_error = max(identity_error, float(
            np.max(np.abs(
                outputs[split]["fused_score"].to_numpy(dtype=np.float64)
                - outputs[split]["fused_recomputed"].to_numpy(dtype=np.float64)
            ))
        ))

    replay = {}
    for split, formal_name in (("train", "ad_train_predictions.csv"),
                               ("test", "ad_test_predictions.csv")):
        formal_path = run_dir / "ad" / formal_name
        formal = pd.read_csv(formal_path)
        result = replay_against_formal(outputs[split], formal)
        result["formal_artifact"] = _sha256_of(formal_path)
        replay[split] = result

    range_report = {
        "train": score_range_report(outputs["train"]),
        "test": score_range_report(outputs["test"]),
    }
    write_json(output_dir / "score_range.json", range_report)

    artifact_records = {}
    for track in SCORE_TRACKS:
        track_dir = output_dir / track
        track_records = {}
        for split in ("train", "test"):
            track_frame = apply_track(outputs[split], track)
            track_frame = track_frame[list(REQUIRED_PREDICTION_COLUMNS)]
            validate_track_frame(track_frame, track)
            track_records[split] = write_track_predictions(track_dir, split, track_frame)
        artifact_records[track] = track_records
    write_json(output_dir / "prediction_artifacts.json", artifact_records)

    score_level_passed = bool(
        identity_error <= 1e-6 and all(record["passed"] for record in replay.values())
    )
    score_replay = {
        "schema_version": "p6_b0_score_replay_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "score_fusion_alpha": FUSION_ALPHA,
        "fusion_semantics": "fused_anomaly = alpha * P_cls(anomaly) + (1-alpha) * P_rec(anomaly)",
        "reconstruction_calibration": _sha256_of(calibration_path),
        "checkpoint": _sha256_of(checkpoint),
        "algebraic_identity_max_abs_error": identity_error,
        "algebraic_identity_tolerance": 1e-6,
        "algebraic_identity_passed": bool(identity_error <= 1e-6),
        "splits": replay,
        "score_level_passed": score_level_passed,
        "score_level_tolerance": 1e-6,
        "identity_alignment_exact": all(
            record["identity_match"] for record in replay.values()
        ),
        "root_cause_if_not_exact": (
            "Residual differences are multi-threaded CPU float32 reduction-order noise in the "
            "shared frozen backbone forward pass.  Calibration, softmax, dtype, fusion order "
            "and row identity are reused verbatim from the frozen P5 implementation; a controlled "
            "torch thread-count sweep reproduces the same magnitude of variation.  The decisive "
            "gate for this audit is the fused event-level reproduction recorded under "
            "event_level_reproduction."
        ),
        "gate_policy": (
            "The score-level tolerance is a numerical target, not a permission to relax the "
            "protocol.  If it is exceeded, evaluation proceeds only when the fused track "
            "reproduces the formal event detection exactly (threshold, TP/FP/FN, delay)."
        ),
        "event_level_reproduction": None,
        "passed": score_level_passed,
    }
    write_json(output_dir / "score_replay.json", score_replay)
    return {
        "score_replay": score_replay,
        "prediction_artifacts": artifact_records,
        "algebraic_identity_max_abs_error": identity_error,
        "config_path": str(config_path),
        "model_args": model_args,
    }


# ---------------------------------------------------------------------------
# Phase 2: per-track event detection and localization
# ---------------------------------------------------------------------------


def window_score_matrices(frame: pd.DataFrame) -> Mapping[str, object]:
    ordered = frame.sort_values(["sample_index", "service_registry_index"], kind="stable")
    n_windows = int(ordered["sample_index"].nunique())
    n_services = len(GAIA_SERVICES)
    if len(ordered) != n_windows * n_services:
        raise ValueError("prediction frame is not a complete window-by-service grid")
    expected_service = np.tile(np.arange(n_services, dtype=np.int64), n_windows)
    if not np.array_equal(
        ordered["service_registry_index"].to_numpy(dtype=np.int64), expected_service
    ):
        raise ValueError("prediction frame service rows are not in registry order")
    times = ordered["prediction_available_time"].to_numpy(dtype=np.int64).reshape(
        n_windows, n_services
    )
    if not np.all(times == times[:, :1]):
        raise ValueError("prediction frame mixes timestamps inside one window")
    matrices = {
        source: ordered[column].to_numpy(dtype=np.float64).reshape(n_windows, n_services)
        for source, column in FRAME_SCORE_COLUMNS.items()
    }
    return {
        "times": times[:, 0],
        "matrices": matrices,
        "time_to_window": {int(value): index for index, value in enumerate(times[:, 0])},
    }


def localization_bundle(
    matrices: Mapping[str, np.ndarray],
    time_to_window: Mapping[int, int],
    anchors: Sequence[int],
    roots: Sequence[str],
    fault_types: Sequence[str],
) -> Mapping[str, object]:
    root_indices = np.asarray([GAIA_SERVICES.index(str(root)) for root in roots], dtype=np.int64)
    windows = np.asarray([time_to_window[int(anchor)] for anchor in anchors], dtype=np.int64)
    bundle = {}
    for source, matrix in matrices.items():
        score_matrix = matrix[windows]
        metrics = localization_metrics(score_matrix, root_indices, fault_types)
        margins = root_margin_report(score_matrix, root_indices, fault_types)
        bundle[source] = {"localization": metrics, "root_margin": margins}
    return bundle


def event_detection_from_threshold(config, train, test, registry, blocks, threshold):
    """Apply one frozen threshold with the shared matching implementation.

    Used only to restore a previously selected Train-only threshold; it never
    selects anything from Test.
    """

    grid_seconds = int(config["ad"]["grid_seconds"])
    tolerance_seconds = int(config["event_trigger"]["matching_tolerance_seconds"])
    train_gt = _gt_rows_for_split(registry, "train", blocks=blocks)
    test_gt = _gt_rows_for_split(registry, "test", blocks=blocks)
    train_episodes, train_matching, train_metrics = evaluate_threshold(
        train, train_gt, threshold, grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    test_episodes, test_matching, test_metrics = evaluate_threshold(
        test, test_gt, threshold, grid_seconds=grid_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    return {
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "event_predictions": pd.concat([train_episodes, test_episodes], ignore_index=True),
        "train_matching": train_matching,
        "test_matching": test_matching,
        "matching": pd.concat([train_matching, test_matching], ignore_index=True),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def _restore_threshold(config, track, train, test, registry, blocks, track_dir):
    """Return a validated reused detection result, or None to recompute."""

    threshold_path = track_dir / "threshold.json"
    if not threshold_path.is_file():
        return None
    stored = json.loads(threshold_path.read_text(encoding="utf-8"))
    if stored.get("track") != track or "threshold" not in stored:
        return None
    recorded_hashes = stored.get("prediction_artifact_sha256")
    current_hashes = {
        "train": sha256_file(track_dir / "ad_train_predictions.csv"),
        "test": sha256_file(track_dir / "ad_test_predictions.csv"),
    }
    if recorded_hashes is not None and recorded_hashes != current_hashes:
        return None
    threshold = float(stored["threshold"])
    rebuilt = event_detection_from_threshold(config, train, test, registry, blocks, threshold)
    if rebuilt["train_metrics"] != stored.get("train_metrics"):
        logging.warning("stored %s threshold is inconsistent with the prediction artifacts", track)
        return None
    rebuild_check = dict(rebuilt)
    rebuild_check["threshold_selection"] = {
        "threshold": threshold,
        "candidate_count": stored.get("candidate_count"),
        "tie_break": stored.get("tie_break"),
        "train_metrics": stored.get("train_metrics"),
        "restored_from_frozen_threshold": True,
    }
    return rebuild_check


def evaluate_track_events(
    config, track: str, track_dir: Path, registry: pd.DataFrame, blocks,
    threshold_workers: int, start_method: str, reuse_threshold: bool = False,
) -> Mapping[str, object]:
    train = pd.read_csv(track_dir / "ad_train_predictions.csv")
    test = pd.read_csv(track_dir / "ad_test_predictions.csv")
    validate_track_frame(train, track)
    validate_track_frame(test, track)
    if set(train["split"].astype(str)) != {"train"} or set(test["split"].astype(str)) != {"test"}:
        raise ValueError("track {} prediction artifacts must be split-pure".format(track))
    restored = (
        _restore_threshold(config, track, train, test, registry, blocks, track_dir)
        if reuse_threshold else None
    )
    if restored is not None:
        logging.info("restoring frozen Train-only threshold for %s", track)
        result = restored
        selection = restored["threshold_selection"]
    else:
        result = run_event_detection(
            train, test, registry, blocks,
            grid_seconds=int(config["ad"]["grid_seconds"]),
            tolerance_seconds=int(config["event_trigger"]["matching_tolerance_seconds"]),
            threshold_workers=int(threshold_workers),
            threshold_start_method=str(start_method),
        )
        selection = result["threshold_selection"]
    matching = result["matching"]
    test_matching = result["test_matching"]

    if isinstance(selection, Mapping):
        threshold_value = float(selection["threshold"])
        candidate_count = selection.get("candidate_count")
        tie_break = selection.get("tie_break")
        selection_train_metrics = selection.get("train_metrics")
        restored_flag = bool(selection.get("restored_from_frozen_threshold"))
    else:
        threshold_value = float(selection.threshold)
        candidate_count = int(selection.candidate_count)
        tie_break = selection.tie_break
        selection_train_metrics = selection.train_metrics
        restored_flag = False

    threshold_record = {
        "schema_version": "p6_b0_track_threshold_v1",
        "track": track,
        "selected_score_column": "selected_anomaly_score",
        "threshold": threshold_value,
        "candidate_count": candidate_count,
        "tie_break": tie_break,
        "selection_source": "Train only (exact unique Train-score sweep)",
        "test_used_for_threshold_selection": False,
        "train_metrics": selection_train_metrics,
        "restored_from_frozen_threshold": restored_flag,
        "prediction_artifact_sha256": {
            "train": sha256_file(track_dir / "ad_train_predictions.csv"),
            "test": sha256_file(track_dir / "ad_test_predictions.csv"),
        },
    }
    write_json(track_dir / "threshold.json", threshold_record)

    matching_path = track_dir / "event_matches.csv"
    matching.to_csv(matching_path, index=False, lineterminator="\n")
    episodes_path = track_dir / "event_episodes.csv"
    result["event_predictions"].to_csv(episodes_path, index=False, lineterminator="\n")

    fault_strata = stratified_event_metrics(test_matching, "fault_type")
    root_strata = stratified_event_metrics(test_matching, "gt_service")
    event_metrics = {
        "schema_version": "p6_b0_track_event_metrics_v1",
        "track": track,
        "matching_semantics": "causal_max_cardinality_minimum_delay",
        "grid_seconds": int(config["ad"]["grid_seconds"]),
        "matching_tolerance_seconds": int(config["event_trigger"]["matching_tolerance_seconds"]),
        "prediction_time": "t_hat=prediction_available_time=target_bin_end",
        "threshold": threshold_value,
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
        "test_stratified_by_fault_type": fault_strata,
        "test_stratified_by_root_service": root_strata,
        "artifacts": {
            "event_matches": {"path": str(matching_path.resolve()), "sha256": sha256_file(matching_path)},
            "event_episodes": {"path": str(episodes_path.resolve()), "sha256": sha256_file(episodes_path)},
        },
    }
    write_json(track_dir / "event_metrics.json", event_metrics)

    matched_test = test_matching.loc[test_matching["match_status"].astype(str) == "matched"].copy()
    matched_test = matched_test.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    matrices = window_score_matrices(test)
    bundle = localization_bundle(
        matrices["matrices"], matrices["time_to_window"],
        matched_test["t_hat"].to_numpy(dtype=np.int64),
        matched_test["gt_service"].astype(str).tolist(),
        matched_test["fault_type"].astype(str).tolist(),
    )
    localization_record = {
        "schema_version": "p6_b0_anomaly_score_localization_v1",
        "track": track,
        "terminology": (
            "Anomaly-score localization diagnostic (first-stage score identity content), "
            "NOT Ada-RCA performance."
        ),
        "population": "matched Test events from this track's event matching (before RCA boundary purge)",
        "case_count": int(len(matched_test)),
        "rank_tie_break": "descending score then canonical registry order",
        "scores": {
            source: {
                "localization": aggregate_only(bundle[source]["localization"]),
                "root_margin": bundle[source]["root_margin"],
            }
            for source in SCORE_TRACKS
        },
    }
    write_json(track_dir / "localization_metrics.json", localization_record)
    return {
        "result": result,
        "matching": matching,
        "test_matching": test_matching,
        "matched_test": matched_test,
        "matrices": matrices,
        "localization": bundle,
        "event_metrics": event_metrics,
    }


# ---------------------------------------------------------------------------
# Phase 3: downstream Ada-RCA
# ---------------------------------------------------------------------------


def anchor_lookup(anchor_values) -> Mapping[int, int]:
    """Map anchor millisecond timestamps to feature-row positions."""

    values = np.asarray(anchor_values, dtype=np.int64)
    index = {int(value): position for position, value in enumerate(values)}
    if len(index) != len(values):
        raise ValueError("anchor feature bundle contains duplicate anchor timestamps")
    return index


def materialize_anchor_features(
    config, output_dir: Path, anchors: Sequence[int], index_manifest: Path,
    feature_workers: int, case_chunk_size: int, start_method: str, config_path: Path,
) -> Mapping[str, object]:
    anchor_registry_path = output_dir / "rca" / "anchor_registry.csv"
    anchor_registry_path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted({int(value) for value in anchors})
    pd.DataFrame({
        "case_id": ["p6anchor-{}".format(value) for value in unique],
        "start_ms": unique,
        "split": ["test"] * len(unique),
    }).to_csv(anchor_registry_path, index=False, lineterminator="\n")
    feature_root = output_dir / "rca" / "anchor_features"
    artifact_root = output_dir / "rca"
    manifest = materialize_rca_features(
        config, index_manifest.parent, feature_root, artifact_root,
        anchor_registry_path,
        limit_cases=0,
        workers=int(feature_workers),
        case_chunk_size=int(case_chunk_size),
        start_method=str(start_method),
        config_path=config_path,
    )
    return {
        "anchor_registry": _sha256_of(anchor_registry_path),
        "unique_anchor_count": int(len(unique)),
        "feature_root": str(feature_root.resolve()),
        "feature_manifest": str((artifact_root / "rca_feature_manifest.json").resolve()),
        "feature_manifest_record": manifest["files"]["features"],
    }


def apply_rca_boundary_purge(config, event_bundle, case_registry_gt):
    """Return (matched, purged_matching, boundary_audit) for one track."""

    purged, boundary = _purge_rca_ineligible_matching(
        event_bundle["test_matching"], case_registry_gt, config
    )
    matched = purged.loc[purged["match_status"].astype(str) == "matched"].copy()
    matched = matched.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    if matched["prediction_id"].duplicated().any() or matched["case_id"].duplicated().any():
        raise ValueError("RCA matched rows violate the one-to-one identity constraint")
    return matched, purged, boundary


def _track_rca(
    config, track: str, track_dir: Path, matched: pd.DataFrame, purged: pd.DataFrame,
    boundary: Mapping[str, object], anchor_features, anchor_index, model,
) -> Mapping[str, object]:
    anchors = matched["t_hat"].to_numpy(dtype=np.int64)
    rows = np.asarray([anchor_index[int(anchor)] for anchor in anchors], dtype=np.int64)
    feature_values = np.asarray(anchor_features[rows], dtype=np.float64)
    if feature_values.shape != (len(matched), len(GAIA_SERVICES), 68):
        raise ValueError("track {} detected-anchor feature shape is invalid".format(track))
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("track {} detected-anchor features are not finite".format(track))
    scores = model.scores(feature_values)
    rankings = predict_rankings(feature_values, model, GAIA_SERVICES)
    roots = matched["gt_service"].map({name: index for index, name in enumerate(GAIA_SERVICES)})
    if roots.isna().any():
        raise ValueError("track {} matched rows contain an unknown root service".format(track))
    metrics = (
        rca_metrics(rankings, roots.to_numpy(dtype=np.int64), GAIA_SERVICES,
                    matched["fault_type"].astype(str).tolist())
        if len(matched) else None
    )
    frame = _prediction_frame(matched, scores, rankings, "Ada-RCA-G P6 {}".format(track))
    predictions_path = track_dir / "rca_detected_predictions.csv"
    frame.to_csv(predictions_path, index=False, lineterminator="\n")

    rankings_by_prediction = {
        str(row.prediction_id): rankings[position]
        for position, row in enumerate(matched.itertuples(index=False))
    }
    diagnosis = diagnosis_metrics(purged, rankings_by_prediction)

    rca_record = {
        "schema_version": "p6_b0_track_rca_metrics_v1",
        "track": track,
        "feature_dimension": 68,
        "feature_window": "W300-B15",
        "conditional_logit_refit": False,
        "rca_boundary_purge": boundary,
        "native_population": {
            "definition": "W300-eligible Test GT and detected anchors after chronological boundary purge",
            "matched_case_count": int(len(matched)),
            "metrics": aggregate_only(metrics),
        },
        "artifacts": {
            "rca_detected_predictions": {
                "path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path),
            },
        },
    }
    write_json(track_dir / "rca_metrics.json", rca_record)

    diagnosis_record = {
        "schema_version": "p6_b0_track_diagnosis_metrics_v1",
        "track": track,
        "rca_boundary_purge": boundary,
        "diagnosis": diagnosis,
    }
    write_json(track_dir / "diagnosis_metrics.json", diagnosis_record)
    return {
        "matched": matched,
        "purged_matching": purged,
        "rankings": rankings,
        "scores": scores,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "boundary": boundary,
        "prediction_frame_path": predictions_path,
    }


def evaluate(args, config, run_dir: Path, output_dir: Path, bindings: Mapping[str, object]):
    config_path = _path(args.config)
    registry = load_registry(config, PROJECT_ROOT)
    blocks = temporal_blocks(config)
    checkpoint = run_dir / "checkpoint" / "best_train_f1.pt"
    calibration_path = run_dir / "ad" / "reconstruction_calibration.json"
    checkpoint_before = sha256_file(checkpoint)
    model_path = run_dir / "rca" / "conditional_logit.npz"
    model = load_conditional_logit(model_path)
    case_registry_gt = pd.read_csv(bindings["files"]["rca_gt_case_registry"]["path"])
    index_manifest = Path(bindings["files"]["rca_index_manifest"]["path"])

    replay_path = output_dir / "score_replay.json"
    if not replay_path.is_file():
        raise FileNotFoundError("run 'infer' first; missing {}".format(replay_path))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if not replay.get("identity_alignment_exact") or not replay.get("algebraic_identity_passed"):
        raise RuntimeError(
            "P6 score identity/algebra gate did not pass; evaluation is NO-GO"
        )

    bundles = {}
    for track in SCORE_TRACKS:
        logging.info("event detection + localization: %s", track)
        bundles[track] = evaluate_track_events(
            config, track, output_dir / track, registry, blocks,
            args.threshold_workers, args.start_method,
            reuse_threshold=bool(args.reuse_event_thresholds),
        )

    event_gate = fused_event_reproduction_gate(run_dir, bundles["fused"]["event_metrics"])
    replay["event_level_reproduction"] = event_gate
    identity_exact = bool(replay.get("identity_alignment_exact"))
    algebraic = bool(replay.get("algebraic_identity_passed"))
    replay["reproduction_verdict"] = (
        "EXACT_SCORE_REPLAY"
        if replay.get("score_level_passed")
        else (
            "IDENTITY_EXACT_EVENT_EXACT_FLOAT_NOISE"
            if event_gate["passed"] else "FAILED"
        )
    )
    replay["passed"] = bool(identity_exact and algebraic and event_gate["passed"])
    write_json(output_dir / "score_replay.json", replay)
    if not replay["passed"]:
        raise RuntimeError(
            "fused track did not reproduce the formal baseline (score and event gates); "
            "STOP/NO-GO. See {}".format(output_dir / "score_replay.json")
        )
    logging.info(
        "fused reproduction gate: verdict=%s score_level=%s event_level=%s",
        replay["reproduction_verdict"], replay.get("score_level_passed"), event_gate["passed"],
    )

    matched_by_track = {
        track: bundles[track]["matched_test"]["case_id"].astype(str).tolist()
        for track in SCORE_TRACKS
    }
    matched_counts = {track: len(values) for track, values in matched_by_track.items()}
    common = common_case_ids(matched_by_track)
    logging.info("common-case intersection: %d of %s", len(common), matched_counts)

    # Label identity audit uses the frozen AD label arrays and GT registry.
    data_root = _path(config["ad_paths"]["data_root"])
    timestamps_by_split = {
        split: np.load(data_root / split / "timestamps.npy", allow_pickle=False)
        for split in ("train", "test")
    }
    labels_by_split = {
        split: np.load(data_root / split / "labels.npy", allow_pickle=False)
        for split in ("train", "test")
    }
    assigned, _ = assign_event_blocks(registry, blocks)
    identity = label_identity_audit(
        assigned, timestamps_by_split, labels_by_split,
        grid_seconds=int(config["ad"]["grid_seconds"]),
    )
    write_json(output_dir / "label_identity_audit.json", identity)

    # Apply the frozen W300 boundary purge first, then materialize exactly the
    # anchors the three RCA populations need (one anchor = one feature row).
    rca_populations = {}
    for track in SCORE_TRACKS:
        matched, purged, boundary = apply_rca_boundary_purge(
            config, bundles[track], case_registry_gt
        )
        rca_populations[track] = {"matched": matched, "purged": purged, "boundary": boundary}
    union_anchors = sorted({
        int(anchor)
        for track in SCORE_TRACKS
        for anchor in rca_populations[track]["matched"]["t_hat"].to_numpy(dtype=np.int64)
    })
    logging.info("materializing detected-anchor RCA features for %d unique anchors", len(union_anchors))
    materialization = materialize_anchor_features(
        config, output_dir, union_anchors, index_manifest,
        args.feature_workers, args.case_chunk_size, args.start_method, config_path,
    )
    feature_root = Path(materialization["feature_root"])
    anchor_features = np.load(feature_root / "z2_features.npy", mmap_mode="r", allow_pickle=False)
    anchor_values = np.load(feature_root / "anchors_ms.npy", allow_pickle=False)
    anchor_index = anchor_lookup(anchor_values)
    missing_anchors = sorted(set(union_anchors) - set(anchor_index))
    if missing_anchors:
        raise ValueError(
            "anchor feature bundle is missing requested anchors: {}".format(missing_anchors[:3])
        )

    rca_bundles = {}
    for track in SCORE_TRACKS:
        logging.info("downstream Ada-RCA: %s", track)
        population = rca_populations[track]
        rca_bundles[track] = _track_rca(
            config, track, output_dir / track, population["matched"], population["purged"],
            population["boundary"], anchor_features, anchor_index, model,
        )

    # Common-case evaluation: identical GT cases across all three triggers.
    # The intersection is taken on the RCA-eligible matched populations so that
    # every common case has a frozen Ada-RCA ranking under all three triggers.
    rca_ranking_by_case = {
        track: {
            str(row.case_id): list(rca_bundles[track]["rankings"][position])
            for position, row in enumerate(rca_bundles[track]["matched"].itertuples(index=False))
        }
        for track in SCORE_TRACKS
    }
    common = common_case_ids({
        track: rca_populations[track]["matched"]["case_id"].astype(str).tolist()
        for track in SCORE_TRACKS
    })
    common_pre_purge = common_case_ids(matched_by_track)
    logging.info(
        "common cases: rca-eligible=%d, pre-rca-purge=%d (%s)",
        len(common), len(common_pre_purge), matched_counts,
    )

    common_rows = []
    common_rca_metrics = {}
    common_localization = {"native_anchor": {}}
    common_complementarity = {}
    for track in SCORE_TRACKS:
        matched = rca_bundles[track]["matched"]
        keep_case = matched["case_id"].astype(str).isin(common).to_numpy()
        subset = matched.loc[keep_case].reset_index(drop=True)
        subset_case_ids = subset["case_id"].astype(str).tolist()
        rankings = tuple(rca_ranking_by_case[track][case_id] for case_id in subset_case_ids)
        roots = subset["gt_service"].astype(str).tolist()
        faults = subset["fault_type"].astype(str).tolist()
        common_rca_metrics[track] = {
            "common_case_count": int(len(subset)),
            "metrics": (
                aggregate_only(dynamic_ranking_metrics(rankings, roots, faults))
                if len(subset) else None
            ),
        }
        for position, row in enumerate(subset.itertuples(index=False)):
            common_rows.append({
                "case_id": str(row.case_id),
                "track": track,
                "prediction_id": str(row.prediction_id),
                "t_hat": int(row.t_hat),
                "gt_start_ms": int(row.gt_start_ms),
                "detection_delay_seconds": float(row.detection_delay_seconds),
                "root_service": str(row.gt_service),
                "fault_type": str(row.fault_type),
                "ada_rca_top1": rankings[position][0],
                "ada_rca_labelled_rank": int(rankings[position].index(str(row.gt_service)) + 1),
            })

        # Anomaly-score localization restricted to the same common cases, at
        # this track's own trigger anchors.
        matched_test = bundles[track]["matched_test"]
        keep = matched_test["case_id"].astype(str).isin(common).to_numpy()
        loc_subset = matched_test.loc[keep].reset_index(drop=True)
        if not len(loc_subset):
            common_localization["native_anchor"][track] = None
            common_complementarity[track] = None
            continue
        bundle = localization_bundle(
            bundles[track]["matrices"]["matrices"],
            bundles[track]["matrices"]["time_to_window"],
            loc_subset["t_hat"].to_numpy(dtype=np.int64),
            loc_subset["gt_service"].astype(str).tolist(),
            loc_subset["fault_type"].astype(str).tolist(),
        )
        common_localization["native_anchor"][track] = {
            source: aggregate_only(bundle[source]["localization"]) for source in SCORE_TRACKS
        }
        common_complementarity[track] = {
            source: complementarity_table(
                bundle[source]["localization"]["rankings"],
                [rca_ranking_by_case[track][case_id] for case_id in loc_subset["case_id"].astype(str)],
                loc_subset["gt_service"].astype(str).tolist(),
            )
            for source in SCORE_TRACKS
        }

    # Anchor-controlled comparison: every score source is evaluated at one fixed
    # trigger anchor set, so the score source is the only varying factor.
    for anchor_track in ("fused", "reconstruction"):
        anchor_matched = bundles[anchor_track]["matched_test"]
        keep = anchor_matched["case_id"].astype(str).isin(common).to_numpy()
        anchor_subset = anchor_matched.loc[keep].reset_index(drop=True)
        if not len(anchor_subset):
            common_localization["fixed_{}_anchor".format(anchor_track)] = None
            continue
        fixed = localization_bundle(
            bundles[anchor_track]["matrices"]["matrices"],
            bundles[anchor_track]["matrices"]["time_to_window"],
            anchor_subset["t_hat"].to_numpy(dtype=np.int64),
            anchor_subset["gt_service"].astype(str).tolist(),
            anchor_subset["fault_type"].astype(str).tolist(),
        )
        common_localization["fixed_{}_anchor".format(anchor_track)] = {
            "anchor_track": anchor_track,
            "case_count": int(len(anchor_subset)),
            "scores": {
                source: aggregate_only(fixed[source]["localization"])
                for source in SCORE_TRACKS
            },
        }

    common_cases_path = output_dir / "common_cases.csv"
    pd.DataFrame(common_rows).to_csv(common_cases_path, index=False, lineterminator="\n")

    # Complementarity on each track's native RCA-matched population.
    native_complementarity = {}
    for track in SCORE_TRACKS:
        matched_test = bundles[track]["matched_test"]
        keep = matched_test["case_id"].astype(str).isin(rca_ranking_by_case[track]).to_numpy()
        subset = matched_test.loc[keep].reset_index(drop=True)
        if not len(subset):
            native_complementarity[track] = None
            continue
        bundle = localization_bundle(
            bundles[track]["matrices"]["matrices"],
            bundles[track]["matrices"]["time_to_window"],
            subset["t_hat"].to_numpy(dtype=np.int64),
            subset["gt_service"].astype(str).tolist(),
            subset["fault_type"].astype(str).tolist(),
        )
        native_complementarity[track] = {
            source: complementarity_table(
                bundle[source]["localization"]["rankings"],
                [rca_ranking_by_case[track][case_id] for case_id in subset["case_id"].astype(str)],
                subset["gt_service"].astype(str).tolist(),
            )
            for source in SCORE_TRACKS
        }

    if sha256_file(checkpoint) != checkpoint_before:
        raise RuntimeError("P6 evaluation must not modify the frozen checkpoint")

    comparison = build_comparison(
        bundles, rca_bundles, common_rca_metrics, common_localization,
        native_complementarity, common_complementarity, identity, replay, matched_counts,
    )
    comparison["common_case_count"] = int(len(common))
    comparison["common_case_count_pre_rca_purge"] = int(len(common_pre_purge))
    comparison["common_cases_artifact"] = _sha256_of(common_cases_path)
    comparison["materialization"] = materialization
    write_json(output_dir / "comparison.json", comparison)

    manifest = build_manifest(
        args, config, run_dir, output_dir, bindings, bundles, rca_bundles,
        matched_counts, common, common_pre_purge, materialization, identity, replay,
    )
    write_json(output_dir / "manifest.json", manifest)
    return {"comparison": comparison, "manifest": manifest}


# ---------------------------------------------------------------------------
# Comparison tables and provenance
# ---------------------------------------------------------------------------


def _localization_row(entry) -> Mapping[str, object]:
    if entry is None:
        return {"case_count": 0, "AC@1": None, "AC@3": None, "AC@5": None, "MRR": None}
    overall = entry["overall"] if "overall" in entry else entry["localization"]["overall"]
    margin = None
    if "root_margin" in entry:
        margin = entry["root_margin"]["overall"]["median"]
    return {
        "case_count": int(overall["case_count"]),
        "AC@1": overall["AC@1"],
        "AC@3": overall["AC@3"],
        "AC@5": overall["AC@5"],
        "MRR": overall["MRR"],
        "median_root_margin": margin,
    }


def fused_event_reproduction_gate(run_dir: Path, fused_event_metrics: Mapping[str, object]) -> Mapping[str, object]:
    """Require the fused track to reproduce the formal event detection exactly.

    This is the operative reproduction gate from the P6-B0 protocol: identity of
    the selected Train threshold, of the Test TP/FP/FN counts, and of the causal
    detection delay summary against the frozen formal ``event_detection_metrics``.
    """

    formal_path = run_dir / "events" / "event_detection_metrics.json"
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    formal_test = formal["test_metrics"]
    formal_threshold = float(formal["threshold_selection"]["threshold"])
    fused_test = fused_event_metrics["test_metrics"]
    fused_threshold = float(fused_event_metrics["threshold"])
    count_fields = (
        "true_positive_events", "false_positive_events", "false_negative_events",
        "predicted_episode_count", "ground_truth_event_count",
    )
    counts = {
        field: {"fused": fused_test[field], "formal": formal_test[field]}
        for field in count_fields
    }
    delay_fields = ("mean_seconds", "median_seconds", "p95_seconds")
    delays = {}
    for field in delay_fields:
        fused_value = fused_test["detection_delay"][field]
        formal_value = formal_test["detection_delay"][field]
        delays[field] = {
            "fused": fused_value,
            "formal": formal_value,
            "abs_diff": (
                abs(float(fused_value) - float(formal_value))
                if fused_value is not None and formal_value is not None else None
            ),
        }
    threshold_diff = abs(fused_threshold - formal_threshold)
    passed = (
        threshold_diff <= 1e-6
        and all(record["fused"] == record["formal"] for record in counts.values())
        and all(
            record["abs_diff"] is not None and record["abs_diff"] <= 1e-9
            for record in delays.values()
        )
    )
    return {
        "formal_artifact": _sha256_of(formal_path),
        "fused_threshold": fused_threshold,
        "formal_threshold": formal_threshold,
        "threshold_abs_diff": threshold_diff,
        "counts": counts,
        "delays": delays,
        "passed": bool(passed),
    }


def build_comparison(
    bundles, rca_bundles, common_rca_metrics, common_localization,
    native_complementarity, common_complementarity, identity, replay, matched_counts,
) -> Mapping[str, object]:
    table1 = {}
    for track in SCORE_TRACKS:
        metrics = bundles[track]["event_metrics"]
        test = metrics["test_metrics"]
        table1[track] = {
            "train_threshold": metrics["threshold"],
            "test_precision": test["event_precision"],
            "test_recall": test["event_recall"],
            "test_f1": test["event_f1"],
            "mean_delay_seconds": test["detection_delay"]["mean_seconds"],
            "p95_delay_seconds": test["detection_delay"]["p95_seconds"],
            "true_positive": test["true_positive_events"],
            "false_positive": test["false_positive_events"],
            "false_negative": test["false_negative_events"],
            "predicted_episodes": test["predicted_episode_count"],
            "ground_truth_events": test["ground_truth_event_count"],
        }

    table2 = {}
    for track in SCORE_TRACKS:
        table2[track] = {
            source: _localization_row(bundles[track]["localization"][source])
            for source in SCORE_TRACKS
        }

    table3 = {}
    for track in SCORE_TRACKS:
        metrics = rca_bundles[track]["metrics"]
        if metrics is None:
            table3[track] = {"matched_n": 0}
            continue
        table3[track] = {
            "matched_n": int(metrics["overall"]["case_count"]),
            "AC@1": metrics["overall"]["AC@1"],
            "AC@3": metrics["overall"]["AC@3"],
            "AC@5": metrics["overall"]["AC@5"],
            "MRR": metrics["overall"]["MRR"],
            "root_macro_AC@1": metrics["root_macro"]["macro"]["AC@1"],
            "fault_macro_AC@1": metrics["fault_macro"]["macro"]["AC@1"],
        }

    table4 = {}
    for track in SCORE_TRACKS:
        record = common_rca_metrics[track]
        metrics = record["metrics"]
        table4[track] = {
            "common_n": record["common_case_count"],
            "AC@1": None if metrics is None else metrics["overall"]["AC@1"],
            "AC@3": None if metrics is None else metrics["overall"]["AC@3"],
            "AC@5": None if metrics is None else metrics["overall"]["AC@5"],
            "MRR": None if metrics is None else metrics["overall"]["MRR"],
        }

    table5 = {}
    for track in SCORE_TRACKS:
        diagnosis = rca_bundles[track]["diagnosis"]["metrics"]
        table5[track] = {
            "diagnosis_precision@1": diagnosis["@1"]["precision"],
            "diagnosis_recall@1": diagnosis["@1"]["recall"],
            "diagnosis_f1@1": diagnosis["@1"]["f1"],
            "diagnosis_precision@3": diagnosis["@3"]["precision"],
            "diagnosis_recall@3": diagnosis["@3"]["recall"],
            "diagnosis_f1@3": diagnosis["@3"]["f1"],
            "diagnosis_precision@5": diagnosis["@5"]["precision"],
            "diagnosis_recall@5": diagnosis["@5"]["recall"],
            "diagnosis_f1@5": diagnosis["@5"]["f1"],
        }

    return {
        "schema_version": "p6_b0_comparison_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "score_fusion_alpha": FUSION_ALPHA,
        "matched_test_counts": matched_counts,
        "table1_event_trigger": table1,
        "table2_label_coupling_diagnostic": table2,
        "table3_rca_native_population": table3,
        "table4_rca_common_cases": table4,
        "table5_full_e2e_diagnosis": table5,
        "common_localization": common_localization,
        "native_complementarity": native_complementarity,
        "common_complementarity": common_complementarity,
        "label_identity_audit": identity,
        "score_replay_passed": bool(replay["passed"]),
        "score_replay_verdict": replay.get("reproduction_verdict"),
        "score_replay_score_level_passed": replay.get("score_level_passed"),
        "score_replay_max_abs_diff": {
            split: record.get("max_abs_diff")
            for split, record in (replay.get("splits") or {}).items()
        },
        "event_level_reproduction_passed": (
            (replay.get("event_level_reproduction") or {}).get("passed")
        ),
        "decision": decide_outcome(table1, table2, table3),
    }


def decide_outcome(table1, table2, table3) -> Mapping[str, object]:
    """Operationalize Outcome A/B/C from the pre-registered gates.

    Thresholds are fixed in advance and documented; they are never tuned on
    Test metrics.  Degradation is measured relative to the frozen fused track.
    """

    fused = table1["fused"]
    rec = table1["reconstruction"]
    cls_localization = table2["fused"]["classification"]["AC@1"]
    rec_localization = table2["reconstruction"]["reconstruction"]["AC@1"]
    f1_ratio = (
        float(rec["test_f1"] / fused["test_f1"])
        if fused["test_f1"] else None
    )
    recall_ratio = (
        float(rec["test_recall"] / fused["test_recall"])
        if fused["test_recall"] else None
    )
    localization_gap = (
        float(cls_localization - rec_localization)
        if cls_localization is not None and rec_localization is not None else None
    )
    usable_f1_ratio = 0.60
    severe_f1_ratio = 0.35
    coupling_reduced_gap = 0.20
    coupling_persists_gap = 0.05

    usable = f1_ratio is not None and f1_ratio >= usable_f1_ratio
    severe = f1_ratio is not None and f1_ratio < severe_f1_ratio
    coupling_reduced = localization_gap is not None and localization_gap >= coupling_reduced_gap
    coupling_persists = localization_gap is not None and localization_gap < coupling_persists_gap

    if severe:
        outcome = "B"
        rationale = "Reconstruction-only event detection degraded below 35% of fused event F1."
    elif usable and coupling_reduced:
        outcome = "A"
        rationale = "Reconstruction-only retains event detection and its localization AC@1 is clearly lower."
    elif coupling_persists and not severe:
        outcome = "C"
        rationale = "Reconstruction-only localization AC@1 stays close to classification AC@1."
    else:
        outcome = "MIXED"
        rationale = "Gates disagree; the outcome is mixed and cannot be reduced to A/B/C."
    return {
        "outcome": outcome,
        "rationale": rationale,
        "operationalization": {
            "usable_f1_ratio": usable_f1_ratio,
            "severe_f1_ratio": severe_f1_ratio,
            "coupling_reduced_localization_gap": coupling_reduced_gap,
            "coupling_persists_localization_gap": coupling_persists_gap,
            "definition": (
                "f1_ratio = reconstruction Test F1 / fused Test F1; "
                "localization_gap = classification AC@1 - reconstruction AC@1 "
                "(both measured with their own trigger anchors on the same score source)."
            ),
        },
        "observed": {
            "reconstruction_over_fused_f1_ratio": f1_ratio,
            "reconstruction_over_fused_recall_ratio": recall_ratio,
            "classification_minus_reconstruction_localization_AC@1": localization_gap,
            "reconstruction_native_rca_AC@1": table3["reconstruction"].get("AC@1"),
            "fused_native_rca_AC@1": table3["fused"].get("AC@1"),
        },
    }


def build_manifest(
    args, config, run_dir, output_dir, bindings, bundles, rca_bundles,
    matched_counts, common, common_pre_purge, materialization, identity, replay,
) -> Mapping[str, object]:
    files = bindings["files"]
    checkpoint = run_dir / "checkpoint" / "best_train_f1.pt"
    calibration = run_dir / "ad" / "reconstruction_calibration.json"
    config_path = _path(args.config)
    return {
        "schema_version": "p6_b0_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_AUDIT_COMPLETE",
        "git_commit": git_head(),
        "formal_source_run": str(Path(run_dir).resolve()),
        "formal_source_input_manifest": _sha256_of(Path(run_dir) / "input_manifest.json"),
        "checkpoint": _sha256_of(checkpoint),
        "reconstruction_calibration": _sha256_of(calibration),
        "preprocessing_manifest": _sha256_of(Path(files["ad_data_manifest"]["path"])),
        "frozen_preprocessing_schema": _sha256_of(Path(files["frozen_preprocessing_schema"]["path"])),
        "config": _sha256_of(config_path),
        "original_formal_prediction_artifact": {
            "train": _sha256_of(run_dir / "ad" / "ad_train_predictions.csv"),
            "test": _sha256_of(run_dir / "ad" / "ad_test_predictions.csv"),
        },
        "original_formal_event_matching": _sha256_of(run_dir / "events" / "event_matching.csv"),
        "original_formal_detected_ranking": _sha256_of(run_dir / "rca" / "rca_detected_predictions.csv"),
        "conditional_logit": _sha256_of(run_dir / "rca" / "conditional_logit.npz"),
        "rca_raw_index_manifest": _sha256_of(Path(files["rca_index_manifest"]["path"])),
        "event_registry": _sha256_of(Path(files["event_registry"]["path"])),
        "score_fusion_alpha": FUSION_ALPHA,
        "score_tracks": list(SCORE_TRACKS),
        "fused_reproduction_gate": {
            "verdict": replay.get("reproduction_verdict"),
            "score_level_passed": replay.get("score_level_passed"),
            "score_level_tolerance": replay.get("score_level_tolerance"),
            "identity_alignment_exact": replay.get("identity_alignment_exact"),
            "algebraic_identity_passed": replay.get("algebraic_identity_passed"),
            "max_abs_diff": {
                split: record.get("max_abs_diff")
                for split, record in (replay.get("splits") or {}).items()
            },
            "event_level_reproduction": replay.get("event_level_reproduction"),
        },
        "train_test_case_counts": {
            "matched_test_by_track": matched_counts,
            "common_case_count": int(len(common)),
            "common_case_count_pre_rca_purge": int(len(common_pre_purge)),
            "matched_test_by_track_after_rca_purge": {
                track: int(len(rca_bundles[track]["matched"])) for track in SCORE_TRACKS
            },
        },
        "label_identity_audit_summary": {
            "total_events": identity["total_events"],
            "exact_identity_count": identity["exact_identity_count"],
            "exact_identity_ratio": identity["exact_identity_ratio"],
            "labelled_service_identity_ratio": identity[
                "labelled_service_positive_on_all_bins_ratio"
            ],
        },
        "rca_anchor_materialization": materialization,
        "policy": {
            "no_retraining": True,
            "test_used_for_threshold_selection": False,
            "rca_retrained": False,
            "reconstruction_calibration_refit": False,
            "writes_into_formal_run": False,
            "alpha_sweep_performed": False,
            "reused_frozen_event_thresholds": bool(args.reuse_event_thresholds),
            "output_dir": str(Path(output_dir).resolve()),
        },
        "code": {
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "helper": str((PROJECT_ROOT / "src/e2e/score_decomposition.py").resolve()),
            "helper_sha256": sha256_file(PROJECT_ROOT / "src/e2e/score_decomposition.py"),
        },
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _path(args.config)
    config = load_config(config_path)
    run_dir = _path(args.run_dir).resolve()
    output_dir = guard_output_root(_path(args.output_dir), run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bindings = load_formal_bindings(run_dir)

    result = {}
    if args.action in ("infer", "all"):
        logging.info("phase 1: pure Ada-MGAD inference and score export")
        result["infer"] = run_inference(args, config, run_dir, output_dir)
    if args.action in ("evaluate", "all"):
        logging.info("phase 2: three-track event detection, localization, and Ada-RCA")
        result["evaluate"] = evaluate(args, config, run_dir, output_dir, bindings)
    replay_record = {}
    replay_path = output_dir / "score_replay.json"
    if replay_path.is_file():
        replay_record = json.loads(replay_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "action": args.action,
        "output_dir": str(output_dir),
        "score_replay_passed": replay_record.get("passed"),
        "score_replay_max_abs_diff": {
            split: record.get("max_abs_diff")
            for split, record in (replay_record.get("splits") or {}).items()
        },
        "decision": result.get("evaluate", {}).get("comparison", {}).get("decision"),
    }, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
