#!/usr/bin/env python3
"""Audit the completed P6-A Ada-MGAD -> Ada-RCA E2E failure.

This is an analysis-only entry point.  It consumes an immutable completed run,
recomputes the requested comparisons, and optionally performs an in-memory
anchor-offset simulation with the already fitted conditional-logit model.  It
never writes model, preprocessing, threshold, matching, feature, or prediction
artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex
from src.e2e.protocol import GAIA_SERVICES, load_config, temporal_blocks
from src.e2e.rca_features import (
    TemporalSpec,
    extract_case_features_from_indicators,
    flatten_features,
)
from src.e2e.rca_model import load_conditional_logit, rank_candidates
from src.e2e.e2e_evaluation import delay_ranking_relationship, parse_ranking_row


DEFAULT_RUN = (
    "experiments/p5/gaia_v2/"
    "gaia-v2-seed42-20260915T181440"
)
DEFAULT_GT_FEATURE_ROOT = "data/p5/v3/rca_features_gt"
DEFAULT_INDEX_MANIFEST = "data/p5/v3/rca_raw_index/index_manifest.json"
DEFAULT_CONFIG = "configs/e2e/gaia_p5_v3_preprocessing_v2.json"

CHANNELS = ("metric", "log", "trace-error", "trace-latency")
BASE_FIELDS = (
    "magnitude", "post_mean", "post_minus_pre", "onset_seconds",
    "onset_missing", "persistence", "coverage", "available",
)
Z2_FIELDS = (
    "pre_z_mean", "post_z_mean", "post_minus_pre", "peak_fraction",
    "centroid", "slope", "mean_adjacent", "fraction_high", "active",
)
FAULT_TYPES = (
    "login_failure", "memory_anomalies", "file_moving",
    "access_permission_denied", "cpu_anomalies", "normal_memory_freed",
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: {}".format(type(value).__name__))


def _row_mapping(row: Mapping[str, object]) -> Mapping[str, object]:
    """Accept a pandas Series, dict, or namedtuple from ``itertuples``."""

    if hasattr(row, "get"):
        return row
    if hasattr(row, "_asdict"):
        return row._asdict()
    raise TypeError("ranking row must provide mapping-like access")


def _ranking(row: Mapping[str, object]) -> Tuple[str, ...]:
    return tuple(parse_ranking_row(_row_mapping(row)))


def _rank(row: Mapping[str, object], service: str) -> int:
    return _ranking(row).index(str(service)) + 1


def _metric_values(ranks: Sequence[int]) -> Mapping[str, object]:
    values = np.asarray(tuple(int(value) for value in ranks), dtype=np.int64)
    if values.size == 0:
        return {
            "case_count": 0,
            "AC@1": None,
            "AC@3": None,
            "AC@5": None,
            "MRR": None,
            "Avg@5": None,
        }
    return {
        "case_count": int(values.size),
        "AC@1": float(np.mean(values <= 1)),
        "AC@3": float(np.mean(values <= 3)),
        "AC@5": float(np.mean(values <= 5)),
        "MRR": float(np.mean(1.0 / values.astype(float))),
        "Avg@5": float(np.mean([
            sum(float(rank <= k) for k in range(1, 6)) / 5.0
            for rank in values
        ])),
    }


def _metrics_from_frame(frame: pd.DataFrame) -> Mapping[str, object]:
    if frame.empty:
        return _metric_values(())
    ranks = [_rank(row, row.gt_service) for row in frame.itertuples(index=False)]
    result = dict(_metric_values(ranks))
    # Avg@5 is the mean of the five cumulative hit indicators; keep the exact
    # evaluator arithmetic instead of deriving it from a rounded display value.
    result["Avg@5"] = float(np.mean([
        sum(float(rank <= k) for k in range(1, 6)) / 5.0
        for rank in ranks
    ]))
    return result


def _metrics_from_ranks(ranks: Sequence[int]) -> Mapping[str, object]:
    ranks = tuple(int(value) for value in ranks)
    result = dict(_metric_values(ranks))
    if ranks:
        result["Avg@5"] = float(np.mean([
            sum(float(rank <= k) for k in range(1, 6)) / 5.0
            for rank in ranks
        ]))
    return result


def _summary(values: Iterable[float]) -> Mapping[str, object]:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p05": None,
                "p25": None, "p75": None, "p95": None, "min": None,
                "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _corr(left: Sequence[float], right: Sequence[float], ranked: bool = False):
    x = np.asarray(tuple(float(value) for value in left), dtype=np.float64)
    y = np.asarray(tuple(float(value) for value in right), dtype=np.float64)
    if len(x) < 2 or len(y) != len(x):
        return None
    if ranked:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def _load_required(run_dir: Path) -> Mapping[str, Path]:
    paths = {
        "run_state": run_dir / "run_state.json",
        "final_report": run_dir / "final_report.md",
        "event_matching": run_dir / "events/event_matching.csv",
        "event_metrics": run_dir / "events/event_detection_metrics.json",
        "ad_predictions": run_dir / "events/ad_event_predictions.csv",
        "oracle": run_dir / "rca/rca_oracle_predictions.csv",
        "detected": run_dir / "rca/rca_detected_predictions.csv",
        "detector_only": run_dir / "rca/detector_only_predictions.csv",
        "e2e_metrics": run_dir / "rca/e2e_diagnosis_metrics.json",
        "layered": run_dir / "rca/e2e_layered_report.json",
        "rca_metrics": run_dir / "rca/rca_metrics.json",
        "feature_manifest": run_dir / "rca/detected/rca_feature_manifest.json",
        "feature_health": run_dir / "rca/detected/rca_feature_health.json",
        "detected_features": run_dir / "rca/detected_features/z2_features.npy",
        "detected_case_ids": run_dir / "rca/detected_features/case_ids.npy",
        "detected_splits": run_dir / "rca/detected_features/splits.npy",
        "detected_anchors": run_dir / "rca/detected_features/anchors_ms.npy",
        "model": run_dir / "rca/conditional_logit.npz",
        "model_meta": run_dir / "rca/conditional_logit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("completed P6 run is missing: {}".format(missing))
    return paths


def _prepare_case_frames(paths: Mapping[str, Path]):
    matching = pd.read_csv(paths["event_matching"])
    oracle = pd.read_csv(paths["oracle"])
    detected = pd.read_csv(paths["detected"])
    detector_only = pd.read_csv(paths["detector_only"])
    for frame in (matching, oracle, detected, detector_only):
        if "case_id" in frame:
            frame["case_id"] = frame["case_id"].astype(str)
        if "prediction_id" in frame:
            frame["prediction_id"] = frame["prediction_id"].astype(str)
    if detected["prediction_id"].duplicated().any() or detected["case_id"].duplicated().any():
        raise ValueError("detected ranking artifact has duplicate identities")
    test = matching.loc[matching["split"].astype(str) == "test"].copy()
    test["prediction_id"] = test["prediction_id"].astype(str)
    matched_rows = test.loc[
        (test["match_status"].astype(str) == "matched")
        & test["prediction_id"].isin(set(detected["prediction_id"]))
    ].copy()
    matched_rows = matched_rows.sort_values(
        ["t_hat", "prediction_id"], kind="stable"
    ).reset_index(drop=True)
    if len(matched_rows) != len(detected):
        raise ValueError(
            "detected rows do not equal matched event rows: {} != {}".format(
                len(detected), len(matched_rows)
            )
        )
    detected_by_prediction = detected.set_index("prediction_id", drop=False)
    for row in matched_rows.itertuples(index=False):
        source = detected_by_prediction.loc[str(row.prediction_id)]
        if str(source.case_id) != str(row.case_id):
            raise ValueError("detected ranking case identity disagrees with matching")
        if int(source.t_hat) != int(row.t_hat):
            raise ValueError("detected ranking anchor disagrees with matching")
        delay = (int(row.t_hat) - int(row.gt_start_ms)) / 1000.0
        if not np.isclose(float(source.detection_delay_seconds), delay, atol=1e-9):
            raise ValueError("detected ranking delay disagrees with matching")
    matched = detected.copy()
    matched["delay_seconds_recomputed"] = (
        matched["t_hat"].astype(np.int64) - matched["gt_start_ms"].astype(np.int64)
    ) / 1000.0
    matched = matched.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    return matching, matched_rows, matched, oracle, detector_only


def _delay_audit(matched: pd.DataFrame) -> Mapping[str, object]:
    # Keep the repository's existing relationship implementation as a
    # reference.  The calculations below extend it with the requested 0/30/60
    # buckets and AC@3/AC@5 while retaining the exact same ranking semantics.
    legacy_reference = delay_ranking_relationship(
        matched.assign(match_status="matched"),
        {
            str(row.prediction_id): _ranking(row)
            for row in matched.itertuples(index=False)
        },
    )
    delays = matched["delay_seconds_recomputed"].to_numpy(dtype=np.float64)
    ranks = np.asarray([_rank(row, row.gt_service) for row in matched.itertuples(index=False)])
    reciprocal = 1.0 / ranks.astype(np.float64)
    buckets = {
        "delay_<0s": delays < 0.0,
        "delay_0_to_30s": (delays >= 0.0) & (delays <= 30.0),
        "delay_30_to_60s": (delays > 30.0) & (delays <= 60.0),
        "delay_>60s": delays > 60.0,
    }
    bucket_results = {}
    for name, mask in buckets.items():
        bucket_results[name] = {
            "case_count": int(mask.sum()),
            "delay_seconds": _summary(delays[mask]),
            **_metrics_from_ranks(ranks[mask].tolist()),
        }
    return {
        "matched_cases": int(len(matched)),
        "delay_definition": "(t_hat - gt_start_ms) / 1000; t_hat is prediction_available_time",
        "delay_seconds": _summary(delays),
        "signed_delay_vs_reciprocal_rank_pearson": _corr(delays, reciprocal),
        "signed_delay_vs_reciprocal_rank_spearman": _corr(delays, reciprocal, ranked=True),
        "absolute_delay_vs_reciprocal_rank_pearson": _corr(np.abs(delays), reciprocal),
        "absolute_delay_vs_reciprocal_rank_spearman": _corr(np.abs(delays), reciprocal, ranked=True),
        "legacy_delay_ranking_relationship": legacy_reference,
        "buckets": bucket_results,
    }


def _transition_audit(matched: pd.DataFrame, oracle: pd.DataFrame) -> Mapping[str, object]:
    oracle = oracle.set_index("case_id", drop=False)
    missing = sorted(set(matched["case_id"]) - set(oracle.index))
    if missing:
        raise ValueError("oracle ranking missing matched cases: {}".format(missing[:3]))
    records = []
    for row in matched.itertuples(index=False):
        oracle_row = oracle.loc[str(row.case_id)]
        root = str(row.gt_service)
        oracle_ranking = _ranking(oracle_row)
        detected_ranking = _ranking(row)
        records.append({
            "case_id": str(row.case_id),
            "gt_service": root,
            "oracle_top1": oracle_ranking[0],
            "detected_top1": detected_ranking[0],
            "oracle_rank": int(oracle_ranking.index(root) + 1),
            "detected_rank": int(detected_ranking.index(root) + 1),
        })
    frame = pd.DataFrame(records)
    oracle_rank = frame["oracle_rank"].to_numpy(dtype=np.int64)
    detected_rank = frame["detected_rank"].to_numpy(dtype=np.int64)
    matrix = {
        str(oracle_value): {
            str(detected_value): int(((oracle_rank == oracle_value) &
                                      (detected_rank == detected_value)).sum())
            for detected_value in range(1, 11)
        }
        for oracle_value in range(1, 11)
    }
    top1_cases = oracle_rank == 1
    top1_transition = {
        "oracle_top1_case_count": int(top1_cases.sum()),
        "top1_to_top1": int((top1_cases & (detected_rank == 1)).sum()),
        "top1_to_top3_only": int((top1_cases & (detected_rank >= 2) &
                                   (detected_rank <= 3)).sum()),
        "top1_to_top5_only": int((top1_cases & (detected_rank >= 4) &
                                   (detected_rank <= 5)).sum()),
        "top1_to_outside_top5": int((top1_cases & (detected_rank > 5)).sum()),
    }
    denominator = int(len(frame))
    oracle_top1_service_retention = float(
        np.mean(frame["oracle_top1"].to_numpy() == frame["detected_top1"].to_numpy())
    ) if denominator else None
    conditional_retention = (
        top1_transition["top1_to_top1"] / top1_transition["oracle_top1_case_count"]
        if top1_transition["oracle_top1_case_count"] else None
    )
    top1_transition["top1_to_top1_rate_given_oracle_top1"] = conditional_retention
    return {
        "matched_cases": denominator,
        "oracle_top1_service_equals_detected_top1_rate": oracle_top1_service_retention,
        "oracle_top1_transition": top1_transition,
        "oracle_rank_to_detected_rank_counts": matrix,
        "detected_rank_distribution_given_oracle_top1": {
            str(value): int((detected_rank[top1_cases] == value).sum())
            for value in range(1, 11)
        },
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _feature_group_indices() -> Mapping[str, np.ndarray]:
    groups: Dict[str, list[int]] = {
        "magnitude": [],
        "temporal": [],
        "persistence": [],
        "mask_channel": [],
    }
    for channel_index in range(len(CHANNELS)):
        base = channel_index * 17
        groups["magnitude"].extend(base + index for index in (0, 1, 2, 8, 9, 10))
        groups["temporal"].extend(base + index for index in (3, 4, 11, 12, 13, 14))
        groups["persistence"].extend(base + index for index in (5, 15))
        groups["mask_channel"].extend(base + index for index in (6, 7, 16))
    for channel_index, channel in enumerate(CHANNELS):
        groups["channel_{}".format(channel)] = np.arange(
            channel_index * 17, (channel_index + 1) * 17, dtype=np.int64
        ).tolist()
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}


def _load_feature_bundle(root: Path):
    features = np.load(root / "z2_features.npy", mmap_mode="r", allow_pickle=False)
    case_ids = np.load(root / "case_ids.npy", allow_pickle=False).astype(str)
    if features.ndim != 3 or features.shape[1:] != (len(GAIA_SERVICES), 68):
        raise ValueError("feature bundle shape is not (cases,10,68): {}".format(features.shape))
    if len(case_ids) != len(features) or len(set(case_ids)) != len(case_ids):
        raise ValueError("feature bundle case IDs are not aligned and unique")
    return features, case_ids


def _feature_drift(
    matched: pd.DataFrame, gt_feature_root: Path, detected_feature_root: Path,
) -> Mapping[str, object]:
    gt_features, gt_ids = _load_feature_bundle(gt_feature_root)
    detected_features, detected_ids = _load_feature_bundle(detected_feature_root)
    gt_index = {case_id: index for index, case_id in enumerate(gt_ids)}
    detected_index = {case_id: index for index, case_id in enumerate(detected_ids)}
    case_ids = matched["case_id"].astype(str).tolist()
    if any(case_id not in gt_index for case_id in case_ids):
        raise ValueError("GT feature bundle misses a matched case")
    if any(case_id not in detected_index for case_id in case_ids):
        raise ValueError("detected feature bundle misses a matched case")
    gt = np.asarray(gt_features[[gt_index[case_id] for case_id in case_ids]], dtype=np.float64)
    detected = np.asarray(
        detected_features[[detected_index[case_id] for case_id in case_ids]], dtype=np.float64
    )
    root_indices = np.asarray([
        GAIA_SERVICES.index(str(value)) for value in matched["gt_service"]
    ], dtype=np.int64)
    overall = []
    root = []
    nonroot = []
    for index, root_index in enumerate(root_indices):
        overall.append(_cosine(gt[index], detected[index]))
        root.append(_cosine(gt[index, root_index], detected[index, root_index]))
        keep = [candidate for candidate in range(len(GAIA_SERVICES)) if candidate != root_index]
        nonroot.append(_cosine(gt[index, keep], detected[index, keep]))
    similarity = {
        "overall_10x68": _summary(overall),
        "root_service_68": _summary(root),
        "non_root_services_9x68": _summary(nonroot),
    }
    groups = _feature_group_indices()
    group_similarity = {}
    group_delta = {}
    for name, indices in groups.items():
        cosine = []
        mean_abs_delta = []
        relative_l2 = []
        for index in range(len(case_ids)):
            left = gt[index, :, indices]
            right = detected[index, :, indices]
            cosine.append(_cosine(left, right))
            delta = np.asarray(left - right, dtype=np.float64)
            mean_abs_delta.append(float(np.mean(np.abs(delta))))
            denominator = float(np.linalg.norm(left))
            relative_l2.append(float(np.linalg.norm(delta) / denominator)
                                if denominator > 0.0 else float("nan"))
        group_similarity[name] = _summary(cosine)
        group_delta[name] = {
            "mean_absolute_delta": _summary(mean_abs_delta),
            "relative_l2_delta": _summary(relative_l2),
        }
    return {
        "matched_cases": len(case_ids),
        "feature_shape_per_case": [len(GAIA_SERVICES), 68],
        "feature_variant": "z2 = concatenate(base[8], morphology[9]) per channel; 4 channels",
        "similarity": similarity,
        "group_similarity": group_similarity,
        "group_delta": group_delta,
        "zero_norm_counts": {
            "overall": int(np.sum(~np.isfinite(np.asarray(overall)))),
            "root_service": int(np.sum(~np.isfinite(np.asarray(root)))),
            "non_root_services": int(np.sum(~np.isfinite(np.asarray(nonroot)))),
        },
    }


def _stratified_metrics(frame: pd.DataFrame, field: str, groups: Sequence[str]):
    result = {}
    for group in groups:
        subset = frame.loc[frame[field].astype(str) == str(group)]
        result[str(group)] = _metrics_from_frame(subset)
    return result


def _event_detection_strata(matching: pd.DataFrame, field: str, groups: Sequence[str]):
    test = matching.loc[matching["split"].astype(str) == "test"].copy()
    gt = test.loc[test["match_status"].astype(str).isin(("matched", "miss"))]
    result = {}
    for group in groups:
        subset = gt.loc[gt[field].astype(str) == str(group)]
        matched = subset["match_status"].astype(str) == "matched"
        total = int(len(subset))
        hit = int(matched.sum())
        result[str(group)] = {
            "ground_truth_events": total,
            "matched_events": hit,
            "missed_events": int(total - hit),
            "event_recall": float(hit / total) if total else None,
        }
    return result


def _stratification(matching: pd.DataFrame, oracle: pd.DataFrame, detected: pd.DataFrame):
    return {
        "population": "same matched Test cases for oracle-vs-detected anchor comparison",
        "fault_type": {
            fault: {
                "case_count": int((detected["fault_type"].astype(str) == fault).sum()),
                "oracle": _stratified_metrics(oracle.loc[oracle["case_id"].isin(set(detected["case_id"]))], "fault_type", (fault,))[fault],
                "detected": _stratified_metrics(detected, "fault_type", (fault,))[fault],
            }
            for fault in FAULT_TYPES
        },
        "root_service": {
            service: {
                "case_count": int((detected["gt_service"].astype(str) == service).sum()),
                "small_n": bool((detected["gt_service"].astype(str) == service).sum() < 30),
                "oracle": _stratified_metrics(oracle.loc[oracle["case_id"].isin(set(detected["case_id"]))], "gt_service", (service,))[service],
                "detected": _stratified_metrics(detected, "gt_service", (service,))[service],
            }
            for service in GAIA_SERVICES
        },
        "event_detection_by_fault_type": _event_detection_strata(matching, "fault_type", FAULT_TYPES),
        "event_detection_by_root_service": _event_detection_strata(matching, "gt_service", GAIA_SERVICES),
    }


_OFFSET_INDEX = None
_OFFSET_SPEC = None
_OFFSET_MODEL = None


def _offset_chunk(task):
    offset_seconds, start, records = task
    ranks = []
    for case_id, anchor_ms, root_index in records:
        indicators = _OFFSET_INDEX.case_indicators(int(anchor_ms), _OFFSET_SPEC)
        feature_set = extract_case_features_from_indicators(
            str(case_id), GAIA_SERVICES, indicators, _OFFSET_SPEC
        )
        values = flatten_features(feature_set, "z2")
        scores = _OFFSET_MODEL.scores(values)
        ranking = rank_candidates(GAIA_SERVICES, scores)
        ranks.append(int(ranking.index(GAIA_SERVICES[int(root_index)]) + 1))
    return {"offset_seconds": int(offset_seconds), "start": int(start), "ranks": ranks}


def _ordered_fork_map(function, tasks, workers):
    """Run analysis tasks after fork-inheriting the already loaded raw index.

    The production preprocessing code deliberately uses spawn for isolation.
    This helper is confined to the read-only audit because loading 1,590 mmap
    series independently in every spawn worker causes multi-gigabyte I/O for
    each worker.  No output is shared with the formal pipeline.
    """

    task_list = tuple(tasks)
    if not task_list:
        return (), {"workers": 0, "task_count": 0, "start_method": "fork"}
    effective = min(int(workers), len(task_list))
    if effective <= 1:
        return tuple(function(task) for task in task_list), {
            "workers": 1, "task_count": len(task_list), "start_method": "serial"
        }
    context = multiprocessing.get_context("fork")
    max_in_flight = min(len(task_list), effective * 2)
    futures = {}
    results = [None] * len(task_list)
    next_index = 0
    with ProcessPoolExecutor(max_workers=effective, mp_context=context) as executor:
        while next_index < max_in_flight:
            futures[executor.submit(function, task_list[next_index])] = next_index
            next_index += 1
        while futures:
            future = next(iter(as_completed(tuple(futures))))
            index = futures.pop(future)
            results[index] = future.result()
            if next_index < len(task_list):
                futures[executor.submit(function, task_list[next_index])] = next_index
                next_index += 1
    return tuple(results), {
        "workers": effective, "task_count": len(task_list), "start_method": "fork",
        "max_in_flight": max_in_flight,
    }


def _offset_simulation(
    matched: pd.DataFrame, config: Mapping[str, object], index_manifest: Path,
    model_path: Path, workers: int, chunk_size: int,
) -> Mapping[str, object]:
    offsets = (0, 15, 30, 45, 60)
    blocks = temporal_blocks(config)
    test_block = next(block for block in blocks if block.name == "test")
    radius_ms = int(config["rca"]["window_seconds"]) * 1000
    records = []
    for row in matched.itertuples(index=False):
        records.append((
            str(row.case_id), int(row.gt_start_ms),
            GAIA_SERVICES.index(str(row.gt_service)),
        ))
    spec = TemporalSpec(
        window_seconds=int(config["rca"]["window_seconds"]),
        bin_seconds=int(config["rca"]["bin_seconds"]),
        pre_bins=int(config["rca"]["pre_bins"]),
        post_bins=int(config["rca"]["post_bins"]),
        n_bins=int(config["rca"]["n_bins"]),
        onset_sentinel=float(config["rca"]["onset_sentinel_seconds"]),
        post_position_denominator=float(config["rca"]["post_position_denominator"]),
    )
    # Load once in the parent; fork workers inherit the mmap-backed index and
    # fitted model without reopening every raw series.
    global _OFFSET_INDEX, _OFFSET_SPEC, _OFFSET_MODEL
    _OFFSET_INDEX = GaiaRcaRawIndex.from_manifest(index_manifest)
    _OFFSET_SPEC = spec
    _OFFSET_MODEL = load_conditional_logit(model_path)
    result_rows = []
    parallel_by_offset = []
    for offset in offsets:
        print("OFFSET_START={}".format(offset), file=sys.stderr, flush=True)
        tasks = []
        for start in range(0, len(records), int(chunk_size)):
            chunk = tuple(
                (case_id, anchor_ms + int(offset) * 1000, root_index)
                for case_id, anchor_ms, root_index in records[start:start + int(chunk_size)]
            )
            tasks.append((offset, start, chunk))
        offset_rows, offset_parallel = _ordered_fork_map(
            _offset_chunk, tasks, max(1, int(workers))
        )
        result_rows.extend(offset_rows)
        parallel_by_offset.append(offset_parallel)
        print("OFFSET_DONE={}".format(offset), file=sys.stderr, flush=True)
    by_offset = {offset: [] for offset in offsets}
    for result in result_rows:
        by_offset[int(result["offset_seconds"])].extend(result["ranks"])
    output = {}
    for offset in offsets:
        anchor_values = np.asarray(
            matched["gt_start_ms"].to_numpy(dtype=np.int64) + int(offset) * 1000,
            dtype=np.int64,
        )
        crossing = [
            not test_block.contains_interval(int(anchor) - radius_ms, int(anchor) + radius_ms)
            for anchor in anchor_values
        ]
        output[str(offset)] = {
            "offset_seconds": int(offset),
            "case_count": len(by_offset[offset]),
            "metrics": _metrics_from_ranks(by_offset[offset]),
            "test_w300_context_crossing_cases": int(sum(crossing)),
            "simulation_scope": "non-formal; all matched cases retained even if shifted context crosses Test boundary",
        }
    return {
        "offsets_seconds": list(offsets),
        "model": "existing run/rca/conditional_logit.npz; no refit",
        "feature_extraction": "existing GaiaRcaRawIndex + extract_case_features_from_indicators; in-memory only",
        "parallel_execution": parallel_by_offset,
        "results": output,
    }


def _comparative_metrics(matched: pd.DataFrame, oracle: pd.DataFrame, detector_only: pd.DataFrame):
    case_ids = set(matched["case_id"].astype(str))
    oracle_subset = oracle.loc[oracle["case_id"].isin(case_ids)].copy()
    detector_subset = detector_only.loc[detector_only["case_id"].isin(case_ids)].copy()
    if len(oracle_subset) != len(matched) or len(detector_subset) != len(matched):
        raise ValueError("comparison ranking artifacts do not cover the matched population")
    return {
        "detector_only": _metrics_from_frame(detector_subset),
        "ada_rca_oracle_gt_anchor": _metrics_from_frame(oracle_subset),
        "ada_rca_detected_anchor": _metrics_from_frame(matched),
    }


def run_audit(args) -> Mapping[str, object]:
    run_dir = _path(args.run_dir).resolve()
    paths = _load_required(run_dir)
    state = json.loads(paths["run_state"].read_text(encoding="utf-8"))
    if str(state.get("status")) != "COMPLETE":
        raise ValueError("P6 audit requires run_state status COMPLETE")
    config = load_config(_path(args.config).resolve())
    matching, matched_rows, matched, oracle, detector_only = _prepare_case_frames(paths)
    gt_feature_root = _path(args.gt_feature_root).resolve()
    detected_feature_root = paths["detected_features"].parent
    result = {
        "schema_version": "p6_e2e_failure_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_status": state.get("status"),
        "run_updated_at": state.get("updated_at"),
        "source_artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
            if key in {
                "event_matching", "event_metrics", "oracle", "detected",
                "detector_only", "e2e_metrics", "layered", "rca_metrics",
                "feature_manifest", "feature_health", "detected_features",
                "detected_case_ids", "model", "model_meta",
            }
        },
        "populations": {
            "matching_test_rows": int((matching["split"].astype(str) == "test").sum()),
            "matched_rows_used": int(len(matched)),
            "oracle_test_rows": int(len(oracle)),
            "detector_only_rows": int(len(detector_only)),
            "gt_events_after_rca_purge": json.loads(
                paths["e2e_metrics"].read_text(encoding="utf-8")
            )["counts"]["ground_truth_events"],
        },
        "audit_1_delay_ranking": _delay_audit(matched),
        "audit_2_oracle_to_detected_transition": _transition_audit(matched, oracle),
        "audit_3_feature_drift": _feature_drift(
            matched, gt_feature_root, detected_feature_root
        ),
        "audit_4_stratification": _stratification(matching, oracle, matched),
        "audit_5_comparison": _comparative_metrics(matched, oracle, detector_only),
        "audit_6_anchor_sensitivity": None,
        "limitations": {
            "labels_not_used_for_feature_or_model_selection": True,
            "formal_training_or_preprocessing_rerun": False,
            "offset_simulation_is_non_formal": True,
            "detected_ranking_artifact_is_final_run_output": True,
        },
    }
    if not args.skip_offset_simulation:
        result["audit_6_anchor_sensitivity"] = _offset_simulation(
            matched, config, _path(args.index_manifest).resolve(), paths["model"],
            args.workers, args.chunk_size,
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gt-feature-root", default=DEFAULT_GT_FEATURE_ROOT)
    parser.add_argument("--index-manifest", default=DEFAULT_INDEX_MANIFEST)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--skip-offset-simulation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise SystemExit("--workers and --chunk-size must be positive")
    result = run_audit(args)
    output = args.output_json
    if output:
        output_path = _path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("AUDIT_JSON={}".format(output_path))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
