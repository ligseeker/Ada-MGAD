"""P6-C1-v2 C2 full-diagnosis layer, failure ledger, strata and latency.

Layer 3 keeps the historical P5 penalty semantics: a matched GT event whose root
service is outside Top-k is both a false positive and a false negative; a missed
GT event is a false negative; a false-alarm episode is a false positive; an
invalid RCA context or a missing ranking counts as both. Stage-1 metrics are
quoted from the frozen P6-C0 evaluation and are never re-optimized or re-run.

Every GT event and predicted episode lands in exactly one ledger category. The
hierarchy is by rank depth: ``SUCCESS_TOP1`` (rank 1), ``ROOT_OUTSIDE_TOP1``
(rank 2-3), ``ROOT_OUTSIDE_TOP3`` (rank 4-5), ``ROOT_OUTSIDE_TOP5`` (rank > 5).
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .c1_v2_shared_data import _display_path, bound_path, read_json, sha256
from .rca_model import GAIA_SERVICES
from .protocol import SUPPORTED_FAULT_TYPES
from .system_trigger import (
    DURATION_STRATUM_LABELS, duration_stratum, onset_density,
)

LEDGER_CATEGORIES = (
    "EVENT_MISSED", "EVENT_FALSE_ALARM", "RCA_CONTEXT_INVALID", "RCA_RANKING_MISSING",
    "ROOT_OUTSIDE_TOP1", "ROOT_OUTSIDE_TOP3", "ROOT_OUTSIDE_TOP5", "SUCCESS_TOP1",
)
RCA_DELAY_SECONDS = 300.0
ONSET_MULTIPLICITY_STRATA = ("1", "2", "3", "4_plus")


def quote_frozen_stage1(protocol: Mapping[str, object]) -> Mapping[str, object]:
    """Layer 1: quote the frozen P6-C0 Stage-1 result without recomputation."""
    path = bound_path(protocol["bindings"]["c0_test_metrics"])
    payload = read_json(path)
    required = ("test_event_metrics", "frozen_validation_threshold", "selected_epoch",
                "pre_registered_gate", "p6_c0_decision")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("frozen C0 test metrics lack fields: {}".format(missing))
    return {
        "stage": "layer_1_stage1_event_detection",
        "source": _display_path(path),
        "source_sha256": sha256(path),
        "recomputed": False,
        "frozen_verdict": str(payload["p6_c0_decision"]),
        "selected_epoch": payload["selected_epoch"],
        "frozen_validation_threshold": payload["frozen_validation_threshold"],
        "pre_registered_gate": payload["pre_registered_gate"],
        "test_event_metrics": payload["test_event_metrics"],
        "test_ground_truth_events": payload.get("test_ground_truth_events"),
        "note": "quoted frozen Stage-1 result; C2 never re-runs the C0 Test detector",
    }


def _ranking_map(scope: pd.DataFrame) -> Mapping[str, object]:
    return scope.set_index("prediction_id")[["scope_status", "ranking_status",
                                             "ranking_b", "ranking_c"]].to_dict("index")


def _coerce_ranking(raw: object) -> object:
    """Decode a locked ranking; return None when the ranking is unusable."""
    try:
        decoded = json.loads(raw) if isinstance(raw, str) and raw else None
    except ValueError:
        return None
    if not isinstance(decoded, list) or len(decoded) != len(GAIA_SERVICES):
        return None
    if set(str(item) for item in decoded) != set(GAIA_SERVICES):
        return None
    return [_ for _ in decoded]


def _category_for(gt_service: str, ranking: Sequence[str]) -> str:
    position = list(ranking).index(gt_service) + 1
    if position == 1:
        return "SUCCESS_TOP1"
    if position <= 3:
        return "ROOT_OUTSIDE_TOP1"
    if position <= 5:
        return "ROOT_OUTSIDE_TOP3"
    return "ROOT_OUTSIDE_TOP5"


def _arm_ledger(arm: str, *, gt_rows: pd.DataFrame, scope_by_prediction: Mapping[str, object]):
    records = []
    counters = {"matched": 0, "misses": 0, "false_alarms": 0, "illegal_context": 0,
                "ranking_failure": 0, "sufficient": 0}
    for row in gt_rows.itertuples(index=False):
        base = {"arm": arm, "side": "gt_event", "case_id": str(row.case_id),
                "prediction_id": None if pd.isna(row.prediction_id) else str(row.prediction_id),
                "gt_service": str(row.gt_service), "fault_type": str(row.fault_type),
                "gt_start_ms": int(row.gt_start_ms), "t_hat": None,
                "rank": None, "category": "EVENT_MISSED"}
        if row.match_status == "miss":
            counters["misses"] += 1
            records.append(base)
            continue
        counters["matched"] += 1
        locked = scope_by_prediction[str(row.prediction_id)]
        base["t_hat"] = int(row.t_hat)
        if locked["scope_status"] != "legal":
            counters["illegal_context"] += 1
            base["category"] = "RCA_CONTEXT_INVALID"
            records.append(base)
            continue
        ranking = _coerce_ranking(locked["ranking_" + arm])
        if ranking is None:
            counters["ranking_failure"] += 1
            base["category"] = "RCA_RANKING_MISSING"
            records.append(base)
            continue
        counters["sufficient"] += 1
        position = list(ranking).index(str(row.gt_service)) + 1
        base["rank"] = int(position)
        base["category"] = _category_for(str(row.gt_service), ranking)
        records.append(base)
    return records, counters


def evaluate_c2_full_diagnosis(*, scope: pd.DataFrame,
                               registry: pd.DataFrame, matching: pd.DataFrame,
                               test_interval_ms: tuple) -> Mapping[str, object]:
    """Layer 3 raw-event ledger and full-diagnosis P/R/F1@1,3,5 for arms B and C."""
    start, end = (int(value) for value in test_interval_ms)
    required_scope = {"prediction_id", "scope_status", "ranking_status", "ranking_b", "ranking_c"}
    if required_scope - set(scope.columns) or scope["prediction_id"].duplicated().any():
        raise ValueError("C2 scope is incomplete or duplicated")
    required_match = {"prediction_id", "match_status", "case_id", "gt_service", "fault_type",
                      "gt_start_ms", "gt_end_ms", "t_hat"}
    if required_match - set(matching.columns):
        raise ValueError("C2 matching lacks required GT columns")
    predicted = matching.loc[matching["match_status"].isin(("matched", "false_alarm"))].copy()
    if set(predicted["prediction_id"].astype(str)) != set(scope["prediction_id"].astype(str)):
        raise ValueError("C2 predicted universe differs from the locked scope")
    gt_rows = matching.loc[matching["match_status"].isin(("matched", "miss"))].copy()
    if gt_rows["case_id"].duplicated().any():
        raise ValueError("C2 GT population contains duplicated cases")
    scope_by_prediction = _ranking_map(scope)
    records = []
    summary = {}
    for arm in ("b", "c"):
        arm_records, counters = _arm_ledger(arm, gt_rows=gt_rows,
                                            scope_by_prediction=scope_by_prediction)
        records.extend(arm_records)
        for row in predicted.loc[predicted["match_status"] == "false_alarm"].itertuples(index=False):
            records.append({"arm": arm, "side": "predicted_episode", "case_id": None,
                            "prediction_id": str(row.prediction_id), "gt_service": None,
                            "fault_type": None, "gt_start_ms": None, "t_hat": int(row.t_hat),
                            "rank": None, "category": "EVENT_FALSE_ALARM"})
            counters["false_alarms"] += 1
        metrics = {}
        matched_records = [record for record in arm_records
                           if record["category"] != "EVENT_MISSED"]
        for k in (1, 3, 5):
            tp = sum(1 for record in matched_records
                     if record["rank"] is not None and int(record["rank"]) <= k)
            undiagnosed = len(matched_records) - tp
            fp = counters["false_alarms"] + undiagnosed
            fn = counters["misses"] + undiagnosed
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics["@{}".format(k)] = {
                "diagnosis_true_positive": int(tp),
                "diagnosis_false_positive": int(fp),
                "diagnosis_false_negative": int(fn),
                "diagnosis_recall_denominator": int(tp + fn),
                "diagnosis_precision_denominator": int(tp + fp),
                "precision": float(precision), "recall": float(recall), "f1": float(f1)}
        ledger = pd.DataFrame(records)
        ledger = ledger.loc[ledger["arm"] == arm]
        categories = {name: int((ledger["category"] == name).sum()) for name in LEDGER_CATEGORIES}
        summary[arm] = {"metrics": metrics, "counters": counters,
                        "ledger_categories": categories,
                        "ledger_closure": int(sum(categories.values())) == int(
                            len(gt_rows) + counters["false_alarms"])}
    ledger = pd.DataFrame(records, columns=("arm", "side", "case_id", "prediction_id",
                                            "gt_service", "fault_type", "gt_start_ms",
                                            "t_hat", "rank", "category"))
    for arm in ("b", "c"):
        if not summary[arm]["ledger_closure"]:
            raise ValueError("C2 failure ledger does not close for arm {}".format(arm))
    return {"summary": {"layer": "layer_3_full_diagnosis",
                        "definition": ("TP@k: matched legal GT with root in Top-k; "
                                       "FP@k: false alarms + matched GT outside Top-k or "
                                       "undiagnosed; FN@k: stage-1 misses + matched GT "
                                       "outside Top-k or undiagnosed"),
                        "gt_population": int(len(gt_rows)),
                        "predicted_episodes": int(len(predicted)),
                        "arms": summary,
                        "latency": latency_summary(gt_rows),
                        "strata": strata_summary(gt_rows=gt_rows, scope_by_prediction=scope_by_prediction,
                                                 registry=registry, interval_ms=(start, end)),
                        "ledger_categories": list(LEDGER_CATEGORIES)},
            "ledger": ledger}


def latency_summary(gt_rows: pd.DataFrame) -> Mapping[str, object]:
    matched = gt_rows.loc[gt_rows["match_status"] == "matched"].copy()
    delays = pd.to_numeric(matched["t_hat"], errors="coerce").to_numpy(dtype=float) \
        - pd.to_numeric(matched["gt_start_ms"], errors="coerce").to_numpy(dtype=float)
    delays = delays / 1000.0
    delays = delays[np.isfinite(delays)]
    if not len(delays):
        return {"detection_latency_seconds": None, "rca_data_ready_latency_seconds": None,
                "final_diagnosis_compute_latency": "UNMEASURED"}
    return {
        "detection_latency_seconds": {"definition": "t_det - GT onset",
                                      "mean": float(np.mean(delays)),
                                      "median": float(np.median(delays)),
                                      "p95": float(np.percentile(delays, 95)),
                                      "n": int(len(delays))},
        "rca_data_ready_latency_seconds": {"definition": "(t_det + 300s) - GT onset",
                                           "mean": float(np.mean(delays) + RCA_DELAY_SECONDS),
                                           "median": float(np.median(delays) + RCA_DELAY_SECONDS),
                                           "p95": float(np.percentile(delays, 95) + RCA_DELAY_SECONDS),
                                           "n": int(len(delays))},
        "final_diagnosis_compute_latency": "UNMEASURED",
    }


def _arm_metric(records: Sequence[Mapping[str, object]], arm: str, k: int):
    tp = sum(1 for record in records
             if record["arm"] == arm and record["rank"] is not None and int(record["rank"]) <= k)
    misses = sum(1 for record in records
                 if record["arm"] == arm and record["category"] == "EVENT_MISSED")
    false_alarms = sum(1 for record in records
                       if record["arm"] == arm and record["category"] == "EVENT_FALSE_ALARM")
    undiagnosed = sum(1 for record in records
                      if record["arm"] == arm and record["side"] == "gt_event"
                      and record["category"] in ("RCA_CONTEXT_INVALID", "RCA_RANKING_MISSING",
                                                 "ROOT_OUTSIDE_TOP1", "ROOT_OUTSIDE_TOP3",
                                                 "ROOT_OUTSIDE_TOP5"))
    fp = false_alarms + undiagnosed
    fn = misses + undiagnosed
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision and recall and precision + recall else 0.0)
    return {"n": int(tp + undiagnosed + misses), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": float(f1)}


def strata_summary(*, gt_rows: pd.DataFrame, scope_by_prediction: Mapping[str, object],
                   registry: pd.DataFrame, interval_ms: tuple) -> Mapping[str, object]:
    """Descriptive strata with group sizes; interpretation only."""
    start, end = (int(value) for value in interval_ms)
    matched = gt_rows.loc[gt_rows["match_status"] == "matched"].copy()
    ranks = {"b": {}, "c": {}}
    for row in matched.itertuples(index=False):
        locked = scope_by_prediction[str(row.prediction_id)]
        for arm in ("b", "c"):
            if locked["scope_status"] != "legal":
                ranks[arm][str(row.prediction_id)] = None
                continue
            ranking = _coerce_ranking(locked["ranking_" + arm])
            ranks[arm][str(row.prediction_id)] = (
                None if ranking is None else list(ranking).index(str(row.gt_service)) + 1)

    def group_metrics(frame: pd.DataFrame) -> Mapping[str, object]:
        result = {"n": int(len(frame))}
        for arm in ("b", "c"):
            values = [ranks[arm].get(str(pid)) for pid in frame["prediction_id"]]
            for k in (1, 3, 5):
                correct = sum(1 for value in values if value is not None and value <= k)
                result["AC@{}_numerator_{}".format(k, arm)] = int(correct)
                result["AC@{}_{}".format(k, arm)] = float(correct / len(frame)) if len(frame) else None
            reciprocal = [1.0 / value if value else 0.0 for value in values]
            result["MRR_{}".format(arm)] = float(np.mean(reciprocal)) if len(frame) else None
        return result

    complete = gt_rows.copy()
    complete["duration_seconds"] = (pd.to_numeric(complete["gt_end_ms"], errors="coerce")
                                    - pd.to_numeric(complete["gt_start_ms"], errors="coerce")) / 1000.0
    complete["duration_stratum"] = duration_stratum(complete["duration_seconds"].to_numpy(dtype=float))
    duration_strata = {}
    for label in DURATION_STRATUM_LABELS:
        group = complete.loc[complete["duration_stratum"] == label]
        metrics = group_metrics(group.loc[group["match_status"] == "matched"])
        metrics["gt_events"] = int(len(group))
        metrics["matched"] = int((group["match_status"] == "matched").sum())
        metrics["missed"] = int((group["match_status"] == "miss").sum())
        duration_strata[str(label)] = metrics

    legal = registry.loc[registry["detector_domain"].astype(bool)].copy()
    density = onset_density(legal, origin_ms=start, grid_seconds=30)
    counts = density.set_index("onset_bin")["onset_count"].to_dict()
    complete["onset_bin"] = ((pd.to_numeric(complete["gt_start_ms"], errors="raise")
                              - start) // 30000).astype(np.int64)
    complete["onset_count"] = complete["onset_bin"].map(counts)
    if complete["onset_count"].isna().any():
        raise ValueError("C2 onset multiplicity join lost a GT event")
    multiplicity = {}
    for label in ONSET_MULTIPLICITY_STRATA:
        if label == "4_plus":
            group = complete.loc[complete["onset_count"] >= 4]
        else:
            group = complete.loc[complete["onset_count"] == int(label)]
        metrics = group_metrics(group.loc[group["match_status"] == "matched"])
        metrics["gt_events"] = int(len(group))
        multiplicity[label] = metrics
    multiplicity["histogram"] = {str(int(key)): int(value)
                                for key, value in complete["onset_count"].value_counts().sort_index().items()}

    fault_strata = {}
    for value in SUPPORTED_FAULT_TYPES:
        group = complete.loc[complete["fault_type"] == value]
        metrics = group_metrics(group.loc[group["match_status"] == "matched"])
        metrics["gt_events"] = int(len(group))
        metrics["small_n_descriptive_only"] = bool(len(group) < 20)
        fault_strata[str(value)] = metrics
    root_strata = {}
    for service in GAIA_SERVICES:
        group = complete.loc[complete["gt_service"] == service]
        metrics = group_metrics(group.loc[group["match_status"] == "matched"])
        metrics["gt_events"] = int(len(group))
        metrics["small_n_descriptive_only"] = bool(len(group) < 20)
        root_strata[str(service)] = metrics
    return {"matched_population": int(len(matched)),
            "duration": duration_strata,
            "onset_multiplicity": multiplicity,
            "fault": fault_strata,
            "root_service": root_strata,
            "use": "interpretation only; never post hoc model selection"}
