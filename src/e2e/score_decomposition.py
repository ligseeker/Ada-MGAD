"""Read-only P6-B0 score decomposition and label-coupling diagnostics.

The frozen P5 event trigger consumes a single fused Ada-MGAD anomaly score:

    fused_anomaly = alpha * P_classification(anomaly)
                    + (1 - alpha) * P_reconstruction(anomaly)

This module separates the two branches at inference time so that a P6 audit
can compare classification-only, current-fused, and reconstruction-only
system triggers *without* retraining, re-fitting calibration, or touching any
frozen artifact.

Everything here is pure computation.  The functions never train, never call
``fit``/``fit_reconstruction_calibration``/``fit_conditional_logit``, and never
write into a formal run directory.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .protocol import GAIA_SERVICES
from .rca_model import rca_metrics


SCORE_TRACKS = ("classification", "fused", "reconstruction")
FUSION_ALPHA = 0.7
REPLAY_TOLERANCE = 1e-6
FRAME_SCORE_COLUMNS = {
    "classification": "classification_score",
    "fused": "fused_score",
    "reconstruction": "reconstruction_score",
}
SMALL_N_THRESHOLD = 30
REGISTRY_ORDER = {service: index for index, service in enumerate(GAIA_SERVICES)}


# ---------------------------------------------------------------------------
# Inference-time component export
# ---------------------------------------------------------------------------


def deduplicate_windows(indices: np.ndarray, *values: np.ndarray):
    """Keep one row per ``sample_index`` (deterministic first occurrence).

    Mirrors the frozen P5 prediction collector so that a P6 replay uses the
    exact same window identity rule.
    """

    order = np.argsort(indices, kind="stable")
    sorted_indices = np.asarray(indices)[order]
    keep = np.ones(len(sorted_indices), dtype=bool)
    if len(keep) > 1:
        keep[1:] = sorted_indices[1:] != sorted_indices[:-1]
    return (sorted_indices[keep],) + tuple(
        np.asarray(value)[order][keep] for value in values
    )


def predict_score_components(system, loader, dataset, calibration) -> Mapping[str, np.ndarray]:
    """Return classification/reconstruction/fused scores for every window.

    The fusion path calls the frozen ``MY._fuse_predict_with_reconstruction``
    and ``MY._reconstruction_energy_to_prob`` methods, so the produced fused
    score is numerically identical to the formal P5 prediction artifact.
    """

    system.model.eval()
    indices_list, cls_list, labels_list, rec_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = system.input2device(batch, system.use_gpu)
            scores, _, reconstruction = system.model(
                batch, evaluate=True, return_eval_aux=True
            )
            indices_list.append(
                batch["sample_index"].reshape(-1).long().cpu().numpy()
            )
            cls_list.append(scores.cpu())
            labels_list.append(batch["groundtruth_real"].cpu().numpy())
            rec_list.append(reconstruction.cpu().numpy())
    if not indices_list:
        raise ValueError("score decomposition loader yielded no batches")
    indices = np.concatenate(indices_list).astype(np.int64)
    cls = torch.concat(cls_list, dim=0).numpy()
    labels = np.concatenate(labels_list, axis=0)
    rec = np.concatenate(rec_list, axis=0)
    indices, cls, labels, rec = deduplicate_windows(indices, cls, labels, rec)
    if len(indices) != len(dataset) or not np.array_equal(indices, np.arange(len(dataset))):
        raise ValueError("score decomposition did not cover every window exactly once")

    cls_tensor = torch.as_tensor(cls, dtype=torch.float32)
    rec_tensor = torch.as_tensor(rec, dtype=torch.float32)
    fused = system._fuse_predict_with_reconstruction(
        cls_tensor, rec_tensor, calibration=calibration
    ).numpy()
    rec_prob = system._reconstruction_energy_to_prob(
        rec_tensor.reshape(-1), calibration=calibration
    ).numpy().reshape(rec.shape)
    return {
        "indices": indices,
        "classification": cls[..., 1].reshape(-1).astype(np.float64),
        "reconstruction": rec_prob.reshape(-1).astype(np.float64),
        "fused": fused[..., 1].reshape(-1).astype(np.float64),
        "node_label": np.argmax(labels, axis=-1).reshape(-1).astype(np.int64),
        "raw_reconstruction_energy": rec.reshape(-1).astype(np.float64),
    }


def decomposition_frame(dataset, components: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build the long ``window x service`` score table for one split."""

    window_count = len(dataset)
    service_count = len(GAIA_SERVICES)
    for key in ("classification", "reconstruction", "fused", "node_label"):
        if len(components[key]) != window_count * service_count:
            raise ValueError("score component {} is not window-by-service".format(key))
    window_bins = int(dataset.window_bins)
    grid_ms = int(dataset.grid_ms)
    timestamps = np.asarray(dataset.timestamps)
    window_indices = np.arange(window_count, dtype=np.int64)
    target_indices = window_indices + window_bins - 1
    target_start = timestamps[target_indices].astype(np.int64)
    frame = pd.DataFrame({
        "split": str(dataset.split),
        "sample_index": np.repeat(window_indices, service_count),
        "window_start_time": np.repeat(
            timestamps[window_indices].astype(np.int64), service_count
        ),
        "window_end_time": np.repeat(
            (timestamps[target_indices].astype(np.int64) + grid_ms), service_count
        ),
        "target_bin_start": np.repeat(target_start, service_count),
        "target_bin_end": np.repeat(target_start + grid_ms, service_count),
        "prediction_available_time": np.repeat(target_start + grid_ms, service_count),
        "prediction_timestamp": np.repeat(target_start + grid_ms, service_count),
        "service": np.tile(np.asarray(GAIA_SERVICES, dtype=object), window_count),
        "service_registry_index": np.tile(
            np.arange(service_count, dtype=np.int64), window_count
        ),
        "node_label": np.asarray(components["node_label"], dtype=np.int64),
        "classification_score": np.asarray(components["classification"], dtype=np.float64),
        "reconstruction_score": np.asarray(components["reconstruction"], dtype=np.float64),
        "fused_score": np.asarray(components["fused"], dtype=np.float64),
    })
    frame["fused_recomputed"] = (
        FUSION_ALPHA * frame["classification_score"]
        + (1.0 - FUSION_ALPHA) * frame["reconstruction_score"]
    )
    return frame


def apply_track(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    """Select one score track as ``selected_anomaly_score``/``anomaly_score``."""

    if track not in SCORE_TRACKS:
        raise ValueError("unknown score track: {}".format(track))
    selected = frame.copy()
    selected["selected_anomaly_score"] = selected[FRAME_SCORE_COLUMNS[track]].to_numpy(
        dtype=np.float64
    )
    # ``anomaly_score`` is the frozen event-detection contract column.
    selected["anomaly_score"] = selected["selected_anomaly_score"]
    selected["binary_prediction"] = (selected["anomaly_score"] >= 0.5).astype(np.int64)
    return selected


def score_range_report(frame: pd.DataFrame) -> Mapping[str, object]:
    """Report finite/range legality for each score source."""

    report = {}
    for track in SCORE_TRACKS:
        column = FRAME_SCORE_COLUMNS[track]
        values = frame[column].to_numpy(dtype=np.float64)
        report[track] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "all_finite": bool(np.isfinite(values).all()),
            "within_unit_interval": bool(
                np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0
            ),
        }
    return report


# ---------------------------------------------------------------------------
# Numerical replay gate
# ---------------------------------------------------------------------------


def replay_against_formal(decomposition: pd.DataFrame, formal: pd.DataFrame) -> Mapping[str, object]:
    """Compare P6 fused scores against the formal P5 anomaly score table."""

    required_new = {
        "split", "sample_index", "service", "prediction_available_time",
        "node_label", "fused_score",
    }
    required_formal = {
        "split", "sample_index", "service", "prediction_available_time",
        "node_label", "anomaly_score",
    }
    missing_new = sorted(required_new - set(decomposition.columns))
    missing_formal = sorted(required_formal - set(formal.columns))
    if missing_new or missing_formal:
        raise ValueError(
            "replay inputs missing columns: new={} formal={}".format(
                missing_new, missing_formal
            )
        )
    left = decomposition[[
        "split", "sample_index", "service", "prediction_available_time",
        "node_label", "fused_score",
    ]].copy()
    right = formal[[
        "split", "sample_index", "service", "prediction_available_time",
        "node_label", "anomaly_score",
    ]].rename(columns={"anomaly_score": "formal_score"})
    keys = ["split", "sample_index", "service"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("replay tables contain duplicate split/sample_index/service rows")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_new", "_formal"), validate="one_to_one")
    unmatched = int(merged["fused_score"].isna().sum() + merged["formal_score"].isna().sum())
    diff = (merged["fused_score"] - merged["formal_score"]).abs()
    time_mismatch = int(
        (merged["prediction_available_time_new"] != merged["prediction_available_time_formal"]).sum()
    )
    label_mismatch = int(
        (merged["node_label_new"] != merged["node_label_formal"]).sum()
    )
    result = {
        "new_rows": int(len(left)),
        "formal_rows": int(len(right)),
        "merged_rows": int(len(merged)),
        "unmatched_rows": unmatched,
        "prediction_available_time_mismatch_rows": time_mismatch,
        "node_label_mismatch_rows": label_mismatch,
        "identity_match": bool(
            len(left) == len(right) and unmatched == 0
            and time_mismatch == 0 and label_mismatch == 0
        ),
        "max_abs_diff": float(diff.max()) if len(diff) and np.isfinite(diff.max()) else None,
        "mean_abs_diff": float(diff.mean()) if len(diff) and np.isfinite(diff.mean()) else None,
        "tolerance": REPLAY_TOLERANCE,
    }
    result["passed"] = bool(
        result["identity_match"]
        and result["max_abs_diff"] is not None
        and result["max_abs_diff"] <= REPLAY_TOLERANCE
    )
    return result


# ---------------------------------------------------------------------------
# Anomaly-score localization diagnostic and root margin
# ---------------------------------------------------------------------------


def rank_services_by_score(score_row: Sequence[float]) -> Tuple[str, ...]:
    """Rank the ten services by descending score with registry-order ties."""

    values = np.asarray(score_row, dtype=np.float64)
    if values.shape != (len(GAIA_SERVICES),) or not np.isfinite(values).all():
        raise ValueError("score row must hold one finite score per canonical service")
    order = sorted(
        range(len(GAIA_SERVICES)),
        key=lambda index: (-float(values[index]), REGISTRY_ORDER[GAIA_SERVICES[index]]),
    )
    return tuple(GAIA_SERVICES[index] for index in order)


def localization_metrics(
    score_matrix: np.ndarray,
    root_indices: Sequence[int],
    fault_types: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """AC@1/3/5, Avg@5, MRR of the *anomaly score itself*.

    This is an anomaly-score localization diagnostic, not RCA performance.
    """

    values = np.asarray(score_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(GAIA_SERVICES):
        raise ValueError("score matrix must have shape (cases, 10)")
    roots = [GAIA_SERVICES[int(index)] for index in np.asarray(root_indices, dtype=np.int64)]
    rankings = tuple(rank_services_by_score(row) for row in values) if len(values) else tuple()
    result = dict(dynamic_ranking_metrics(rankings, roots, fault_types))
    result["rankings"] = [list(ranking) for ranking in rankings]
    return result


def _margin_statistics(values: np.ndarray) -> Mapping[str, object]:
    if len(values) == 0:
        return {
            "case_count": 0, "mean": None, "median": None, "q25": None,
            "q75": None, "fraction_margin_positive": None,
        }
    return {
        "case_count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "fraction_margin_positive": float(np.mean(values > 0.0)),
    }


def root_margin_report(
    score_matrix: np.ndarray,
    root_indices: Sequence[int],
    fault_types: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Root-vs-best-non-root margin for one score source."""

    values = np.asarray(score_matrix, dtype=np.float64)
    roots = np.asarray(root_indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(GAIA_SERVICES) or len(values) != len(roots):
        raise ValueError("root margin inputs are not aligned")
    if len(values) == 0:
        return {"overall": _margin_statistics(np.empty(0)), "by_fault_type": {}, "by_root_service": {}}
    margins = np.empty(len(values), dtype=np.float64)
    for position, root in enumerate(roots):
        row = values[position]
        others = np.delete(row, int(root))
        margins[position] = float(row[int(root)] - np.max(others))
    report = {
        "overall": _margin_statistics(margins),
        "by_root_service": {},
        "by_fault_type": {},
    }
    root_names = [GAIA_SERVICES[int(root)] for root in roots]
    for field, labels, target in (
        ("by_root_service", root_names, report["by_root_service"]),
        (
            "by_fault_type",
            None if fault_types is None else [str(value) for value in fault_types],
            report["by_fault_type"],
        ),
    ):
        if labels is None:
            continue
        groups: Dict[str, list] = {}
        for label, margin in zip(labels, margins):
            groups.setdefault(label, []).append(float(margin))
        for label in sorted(groups):
            target[label] = _margin_statistics(np.asarray(groups[label], dtype=np.float64))
    return report


# ---------------------------------------------------------------------------
# Label identity audit
# ---------------------------------------------------------------------------


def label_identity_audit(
    events: pd.DataFrame,
    timestamps_by_split: Mapping[str, np.ndarray],
    labels_by_split: Mapping[str, np.ndarray],
    grid_seconds: int = 30,
) -> Mapping[str, object]:
    """Quantify the AD node label <-> RCA root label identity overlap.

    ``events`` must carry ``case_id, service, start_ms, end_ms, split``.  AD
    node labels are re-derived from the frozen arrays with the same half-open
    overlap rule used by preprocessing, and the audit reports how often an
    event's AD positive service set is exactly its own labelled service.
    """

    required = {"case_id", "service", "start_ms", "end_ms", "split"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError("label identity audit requires columns: {}".format(missing))
    step_ms = int(grid_seconds) * 1000
    service_index = {service: index for index, service in enumerate(GAIA_SERVICES)}

    total = 0
    exact_identity = 0
    rasterization_mismatch = 0
    per_bin_event_count: Dict[Tuple[str, int], int] = {}
    per_bin_service_set: Dict[Tuple[str, int], set] = {}
    per_service_bin_count: Dict[Tuple[str, str, int], int] = {}

    for row in events.itertuples(index=False):
        split = str(row.split)
        if split not in timestamps_by_split:
            raise ValueError("event split {} has no frozen timestamps".format(split))
        timestamps = np.asarray(timestamps_by_split[split], dtype=np.int64)
        labels = np.asarray(labels_by_split[split])
        if int(row.end_ms) <= int(row.start_ms):
            raise ValueError("label identity audit requires positive half-open intervals")
        service = str(row.service)
        if service not in service_index:
            raise ValueError("event service is outside the canonical registry")
        column = service_index[service]
        left = int(np.searchsorted(timestamps, int(row.start_ms) - step_ms, side="right"))
        right = int(np.searchsorted(timestamps, int(row.end_ms), side="left"))
        total += 1
        bins = list(range(left, right))
        if any(int(labels[bin_index, column]) != 1 for bin_index in bins):
            rasterization_mismatch += 1
        identity_holds = True
        for bin_index in bins:
            positive = set(
                GAIA_SERVICES[index]
                for index in np.flatnonzero(np.asarray(labels[bin_index]) == 1)
            )
            key = (split, bin_index)
            per_bin_event_count[key] = per_bin_event_count.get(key, 0) + 1
            per_bin_service_set.setdefault(key, set()).add(service)
            per_service_key = (split, service, bin_index)
            per_service_bin_count[per_service_key] = per_service_bin_count.get(per_service_key, 0) + 1
            if positive != {service}:
                identity_holds = False
        if identity_holds:
            exact_identity += 1

    bins_single_event = sum(1 for count in per_bin_event_count.values() if count == 1)
    bins_multi_event = sum(1 for count in per_bin_event_count.values() if count > 1)
    bins_multi_root = sum(1 for services in per_bin_service_set.values() if len(services) > 1)
    service_bin_multi_event = sum(1 for count in per_service_bin_count.values() if count > 1)
    return {
        "definition": (
            "For every GT injection, the AD positive node set on each overlapped 30s bin "
            "is compared with the RCA labelled root service identity.  Both labels come "
            "from the same frozen registry service column, so the identity is structural; "
            "this audit quantifies it and measures bin-level multi-event ambiguity."
        ),
        "total_events": int(total),
        "labelled_service_positive_on_all_bins_count": int(total - rasterization_mismatch),
        "labelled_service_positive_on_all_bins_ratio": (
            float((total - rasterization_mismatch) / total) if total else None
        ),
        "exact_identity_count": int(exact_identity),
        "exact_identity_ratio": float(exact_identity / total) if total else None,
        "exact_identity_definition": (
            "every overlapped bin has AD positive set exactly equal to the event service"
        ),
        "rasterization_mismatch_events": int(rasterization_mismatch),
        "bins_covered": int(len(per_bin_event_count)),
        "bins_single_event": int(bins_single_event),
        "bins_multi_event": int(bins_multi_event),
        "bins_multi_root_service": int(bins_multi_root),
        "service_bin_multi_event": int(service_bin_multi_event),
        "ad_label_source_column": "node_label in ad_{train,test}_predictions.csv (registry-service rasterization)",
        "rca_label_source_column": "gt_service / registry service (same registry)",
    }


# ---------------------------------------------------------------------------
# Common-case population and complementarity
# ---------------------------------------------------------------------------


def common_case_ids(matched_by_track: Mapping[str, Iterable[str]]) -> set:
    """Return the GT case intersection across all supplied tracks."""

    sets = []
    for track in SCORE_TRACKS:
        if track not in matched_by_track:
            raise ValueError("matched cases missing track {}".format(track))
        sets.append(set(str(value) for value in matched_by_track[track]))
    common = sets[0]
    for value in sets[1:]:
        common = common & value
    return common


def complementarity_table(
    score_rankings: Sequence[Sequence[str]],
    rca_rankings: Sequence[Sequence[str]],
    roots: Sequence[str],
) -> Mapping[str, object]:
    """2x2 anomaly-score Top1 vs Ada-RCA Top1 contingency."""

    if not (len(score_rankings) == len(rca_rankings) == len(roots)):
        raise ValueError("complementarity inputs are not aligned")
    both_correct = score_only = rca_only = both_wrong = 0
    for position, root in enumerate(roots):
        score_hit = tuple(score_rankings[position])[0] == str(root)
        rca_hit = tuple(rca_rankings[position])[0] == str(root)
        if score_hit and rca_hit:
            both_correct += 1
        elif score_hit and not rca_hit:
            score_only += 1
        elif rca_hit and not score_hit:
            rca_only += 1
        else:
            both_wrong += 1
    total = len(roots)
    score_wrong = total - both_correct - score_only
    rca_wrong = total - both_correct - rca_only
    return {
        "case_count": int(total),
        "both_correct": int(both_correct),
        "score_correct_rca_wrong": int(score_only),
        "score_wrong_rca_correct": int(rca_only),
        "both_wrong": int(both_wrong),
        "probability_rca_correct_given_score_wrong": (
            float(rca_only / score_wrong) if score_wrong else None
        ),
        "probability_rca_wrong_given_score_correct": (
            float(score_only / (both_correct + score_only))
            if (both_correct + score_only) else None
        ),
        "score_top1_accuracy": float((both_correct + score_only) / total) if total else None,
        "rca_top1_accuracy": float((both_correct + rca_only) / total) if total else None,
    }


# ---------------------------------------------------------------------------
# Stratified event metrics
# ---------------------------------------------------------------------------


def stratified_event_metrics(
    matching: pd.DataFrame,
    field: str,
    small_n_threshold: int = SMALL_N_THRESHOLD,
) -> Mapping[str, object]:
    """case_count / TP / FN / Recall per fault_type or root service."""

    if field not in matching.columns:
        raise ValueError("matching table lacks field {}".format(field))
    if "match_status" not in matching.columns:
        raise ValueError("matching table lacks match_status")
    frame = matching.loc[matching["match_status"].astype(str).isin(("matched", "miss"))].copy()
    frame = frame.loc[frame[field].notna()]
    groups: Dict[str, Dict[str, int]] = {}
    for row in frame.itertuples(index=False):
        name = str(getattr(row, field))
        record = groups.setdefault(name, {"case_count": 0, "true_positive": 0, "false_negative": 0})
        record["case_count"] += 1
        if str(row.match_status) == "matched":
            record["true_positive"] += 1
        else:
            record["false_negative"] += 1
    by_group = {}
    for name in sorted(groups):
        record = groups[name]
        by_group[name] = {
            **record,
            "recall": (
                float(record["true_positive"] / record["case_count"])
                if record["case_count"] else None
            ),
            "small_n": bool(record["case_count"] < int(small_n_threshold)),
        }
    return {
        "field": field,
        "small_n_threshold": int(small_n_threshold),
        "group_count": len(by_group),
        "by_group": by_group,
    }


def aggregate_only(metrics):
    """Drop per-case detail from a metrics mapping before serialization.

    ``rankings`` and ``case_metrics`` are needed in memory (rank aggregation is
    computed first) but are not part of the published aggregate record.
    """

    if not isinstance(metrics, Mapping):
        return metrics
    return {
        key: value for key, value in metrics.items()
        if key not in ("rankings", "case_metrics")
    }


def dynamic_ranking_metrics(
    rankings: Sequence[Sequence[str]],
    roots: Sequence[str],
    fault_types: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Ranking metrics from already-computed service rankings (names)."""

    roots = [str(value) for value in roots]
    if not rankings:
        empty = {metric: None for metric in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")}
        return {
            "overall": {"case_count": 0, **empty},
            "root_macro": {"by_group": {}, "group_count": 0, "macro": empty.copy()},
            "fault_macro": {"by_group": {}, "group_count": 0, "macro": empty.copy()},
            "case_metrics": [],
            "status": "NO_CASES",
        }
    root_indices = np.asarray([REGISTRY_ORDER[root] for root in roots], dtype=np.int64)
    normalized = tuple(tuple(str(item) for item in ranking) for ranking in rankings)
    return rca_metrics(
        normalized,
        root_indices,
        GAIA_SERVICES,
        None if fault_types is None else [str(value) for value in fault_types],
    )
