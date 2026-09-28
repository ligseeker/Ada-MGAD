"""Post-lock paired C1 RCA evaluation on the reused C0 Test population."""

from __future__ import annotations

import json
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .rca_model import GAIA_SERVICES
from .event_detection import match_events
from .c1_common_cohort import extract_c1_anchor_features
from .rca_model import ConditionalLogitFit, rank_candidates
from .protocol import SUPPORTED_FAULT_TYPES


def _ranking(value) -> Optional[tuple]:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        return None
    sequence = tuple(decoded)
    if len(sequence) != len(GAIA_SERVICES) or set(sequence) != set(GAIA_SERVICES):
        return None
    return sequence


def _case_scores(ranking: Optional[Sequence[str]], root: str) -> Mapping[str, float]:
    if ranking is None:
        return {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "MRR": 0.0}
    position = list(ranking).index(root) + 1
    return {"AC@1": float(position == 1), "AC@3": float(position <= 3),
            "AC@5": float(position <= 5), "MRR": 1.0 / float(position)}


def _strata(cases: pd.DataFrame, field: str, universe: Sequence[str]) -> Mapping[str, object]:
    result = {}
    qualified = []
    for value in universe:
        group = cases.loc[cases[field] == value]
        n = len(group)
        item = {"n": int(n), "qualified_for_macro": bool(n >= 20)}
        for metric in ("AC@1", "AC@3", "AC@5", "MRR"):
            for arm in ("b", "c"):
                key = "{}_{}".format(metric, arm)
                item[key] = float(group[key].sum() / n) if n else None
            item["{}_delta".format(metric)] = (item["{}_c".format(metric)] - item["{}_b".format(metric)]) if n else None
        result[str(value)] = item
        if n >= 20:
            qualified.append(item)
    macro = {}
    for metric in ("AC@1", "AC@3", "AC@5", "MRR"):
        for suffix in ("b", "c", "delta"):
            key = "{}_{}".format(metric, suffix)
            macro[key] = sum(float(row[key]) for row in qualified) / len(qualified) if qualified else None
    return {"by_group": result, "qualified_group_count": len(qualified),
            "undefined_groups": [str(name) for name, row in result.items() if row["n"] == 0],
            "small_descriptive_groups": [str(name) for name, row in result.items() if 0 < row["n"] < 20],
            "macro": macro}


def evaluate_c1_prediction_lock(*, scope: pd.DataFrame, matching: pd.DataFrame,
                                 bootstrap_replicates: int = 10000,
                                 bootstrap_seed: int = 42) -> Mapping[str, object]:
    """Use only the locked scope and evaluator-only Test matching/GT."""
    if bootstrap_replicates != 10000 or bootstrap_seed != 42:
        raise ValueError("C1 bootstrap differs from G2 lock")
    required_scope = {"prediction_id", "t_hat", "scope_status", "ranking_status", "ranking_b", "ranking_c"}
    if required_scope - set(scope.columns) or scope["prediction_id"].duplicated().any():
        raise ValueError("C1 locked Test scope is incomplete or duplicated")
    if (not scope["scope_status"].isin(("legal", "illegal_context")).all()
            or not scope["ranking_status"].isin(("complete", "failed", "not_applicable")).all()
            or not scope.loc[scope["ranking_status"] == "failed", "scope_status"].eq("legal").all()
            or not scope.loc[scope["ranking_status"] == "not_applicable", "scope_status"].eq("illegal_context").all()):
        raise ValueError("C1 locked Test context/ranking statuses drift")
    required_match = {"prediction_id", "match_status", "case_id", "gt_service", "fault_type", "gt_start_ms"}
    if required_match - set(matching.columns):
        raise ValueError("C1 evaluator Test matching lacks required GT fields")
    predicted = matching.loc[matching["match_status"].isin(("matched", "false_alarm"))].copy()
    if (predicted["prediction_id"].duplicated().any()
            or set(predicted["prediction_id"].astype(str)) != set(scope["prediction_id"].astype(str))):
        raise ValueError("C1 locked episode universe differs from Test matching")
    matched = predicted.loc[predicted["match_status"] == "matched"].copy()
    joined = matched.merge(scope, on="prediction_id", how="left", validate="one_to_one",
                           suffixes=("_matching", "_lock"))
    if not np.array_equal(joined["t_hat_matching"].to_numpy(dtype=np.int64),
                          joined["t_hat_lock"].to_numpy(dtype=np.int64)):
        raise ValueError("C1 locked episode anchors differ from Test matching")
    joined = joined.loc[joined["scope_status"] == "legal"].copy()
    if joined.empty or joined["case_id"].duplicated().any():
        raise ValueError("C1 primary locked legal matched-case scope is empty or duplicated")
    if not joined["gt_service"].isin(GAIA_SERVICES).all():
        raise ValueError("C1 Test root service is outside canonical candidates")
    if not joined["fault_type"].isin(SUPPORTED_FAULT_TYPES).all():
        raise ValueError("C1 Test fault type is outside frozen GAIA faults")
    scores = {"b": [], "c": []}
    for row in joined.itertuples(index=False):
        for arm in ("b", "c"):
            scores[arm].append(_case_scores(_ranking(getattr(row, "ranking_" + arm)), str(row.gt_service)))
    for arm in ("b", "c"):
        for metric in ("AC@1", "AC@3", "AC@5", "MRR"):
            joined["{}_{}".format(metric, arm)] = [item[metric] for item in scores[arm]]
    n = len(joined)
    primary = {"n": n,
               "correct_b": int(joined["AC@1_b"].sum()),
               "correct_c": int(joined["AC@1_c"].sum())}
    primary["delta_c_minus_b"] = (primary["correct_c"] - primary["correct_b"]) / n
    primary["paired_transitions"] = {
        "both_correct": int(((joined["AC@1_b"] == 1) & (joined["AC@1_c"] == 1)).sum()),
        "b_only_correct": int(((joined["AC@1_b"] == 1) & (joined["AC@1_c"] == 0)).sum()),
        "c_only_correct": int(((joined["AC@1_b"] == 0) & (joined["AC@1_c"] == 1)).sum()),
        "both_incorrect": int(((joined["AC@1_b"] == 0) & (joined["AC@1_c"] == 0)).sum()),
    }
    joined["utc_onset_day"] = np.floor_divide(
        pd.to_numeric(joined["gt_start_ms"], errors="raise").to_numpy(dtype=np.int64), 86400000)
    days = np.sort(joined["utc_onset_day"].unique())
    day_n = np.asarray([int((joined["utc_onset_day"] == day).sum()) for day in days])
    day_delta = np.asarray([int((joined.loc[joined["utc_onset_day"] == day, "AC@1_c"]
                                - joined.loc[joined["utc_onset_day"] == day, "AC@1_b"]).sum())
                            for day in days])
    rng = np.random.default_rng(bootstrap_seed)
    resampled = rng.integers(0, len(days), size=(bootstrap_replicates, len(days)))
    replicates = day_delta[resampled].sum(axis=1) / day_n[resampled].sum(axis=1)
    primary["onset_day_cluster_bootstrap_95_percentile"] = {
        "replicates": bootstrap_replicates, "seed": bootstrap_seed,
        "cluster_days": len(days),
        "lower": float(np.percentile(replicates, 2.5)),
        "upper": float(np.percentile(replicates, 97.5))}
    secondary = {}
    for metric in ("AC@3", "AC@5", "MRR"):
        sum_b = float(joined["{}_b".format(metric)].sum())
        sum_c = float(joined["{}_c".format(metric)].sum())
        secondary[metric] = {"n": n, "sum_b": sum_b, "sum_c": sum_c,
                             "mean_b": sum_b / n, "mean_c": sum_c / n,
                             "delta_c_minus_b": (sum_c - sum_b) / n}
    root = _strata(joined, "gt_service", GAIA_SERVICES)
    fault = _strata(joined, "fault_type", SUPPORTED_FAULT_TYPES)
    detail_columns = ("prediction_id", "case_id", "gt_service", "fault_type", "gt_start_ms",
                      "t_hat_lock", "scope_status", "AC@1_b", "AC@1_c", "AC@3_b", "AC@3_c",
                      "AC@5_b", "AC@5_c", "MRR_b", "MRR_c", "utc_onset_day")
    return {"primary": primary, "secondary": secondary,
            "root_strata": root, "fault_strata": fault,
            "scope": {"all_locked_episodes": int(len(scope)),
                      "legal_locked_episodes": int((scope["scope_status"] == "legal").sum()),
                      "legal_ranking_failures": int((scope["ranking_status"] == "failed").sum()),
                      "matched_legal_cases": n,
                      "false_alarms": int((matching["match_status"] == "false_alarm").sum()),
                      "misses": int((matching["match_status"] == "miss").sum())},
            "case_details": joined[list(detail_columns)].to_dict(orient="records")}


def evaluate_c2_raw_failure(*, episodes: pd.DataFrame, scope: pd.DataFrame,
                            registry: pd.DataFrame, test_interval_ms: tuple):
    """Post-lock raw-GT failure ledger; final diagnosis wall time is unmeasured."""
    start, end = (int(value) for value in test_interval_ms)
    required_gt = {"case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"}
    if required_gt - set(registry.columns):
        raise ValueError("C2 raw GT registry lacks event fields")
    events = registry.copy()
    intersecting = events.loc[(events["start_ms"] < end) & (events["end_ms"] > start)]
    complete = intersecting.loc[(intersecting["start_ms"] >= start)
                                & (intersecting["end_ms"] <= end)].copy()
    boundary_events = intersecting.loc[~intersecting["case_id"].isin(complete["case_id"])].copy()
    boundary = len(boundary_events)
    raw_matching = match_events(episodes, complete, tolerance_seconds=60)
    paired = raw_matching.loc[raw_matching["match_status"] == "matched"].merge(
        scope, on="prediction_id", how="left", validate="one_to_one")
    if len(paired) != int((raw_matching["match_status"] == "matched").sum()):
        raise ValueError("C2 raw matched episode scope drift")
    diagnoses = {"correct": 0, "wrong_rank": 0,
                 "illegal_context": 0, "ranking_failure": 0}
    detector_delays = []
    data_ready_delays = []
    diagnosis_by_prediction = {}
    ready_by_prediction = {}
    for row in paired.itertuples(index=False):
        delay = (int(row.t_hat_x) - int(row.gt_start_ms)) / 1000.0
        detector_delays.append(delay)
        if row.scope_status != "legal":
            diagnoses["illegal_context"] += 1
            diagnosis_by_prediction[str(row.prediction_id)] = "illegal_context"
            continue
        data_ready_delays.append(delay + 300.0)
        ready_by_prediction[str(row.prediction_id)] = delay + 300.0
        ranking = _ranking(row.ranking_c)
        if ranking is None:
            diagnoses["ranking_failure"] += 1
            diagnosis_by_prediction[str(row.prediction_id)] = "ranking_failure"
            continue
        if ranking[0] == str(row.gt_service):
            diagnoses["correct"] += 1
            diagnosis_by_prediction[str(row.prediction_id)] = "correct"
        else:
            diagnoses["wrong_rank"] += 1
            diagnosis_by_prediction[str(row.prediction_id)] = "wrong_rank"
    raw_matching = raw_matching.copy()
    scope_by_prediction = scope.set_index("prediction_id")
    raw_matching["context_status"] = raw_matching["prediction_id"].map(
        scope_by_prediction["scope_status"])
    raw_matching["ranking_status"] = raw_matching["prediction_id"].map(
        scope_by_prediction["ranking_status"])
    raw_matching["diagnosis_status"] = raw_matching["prediction_id"].map(
        diagnosis_by_prediction)
    raw_matching.loc[raw_matching["match_status"] == "miss", "diagnosis_status"] = "detector_miss"
    raw_matching.loc[raw_matching["match_status"] == "false_alarm", "diagnosis_status"] = "false_alarm"
    raw_matching["rca_data_ready_delay_seconds"] = raw_matching["prediction_id"].map(
        ready_by_prediction)
    raw_matching["final_diagnosis_wall_time_ms"] = np.nan
    if boundary:
        boundary_rows = pd.DataFrame([{
            "split": "test", "prediction_id": None, "t_hat": None,
            "match_status": "boundary_gt", "case_id": str(row.case_id),
            "source_index": int(row.source_index), "gt_service": str(row.service),
            "fault_type": str(row.fault_type), "gt_start_ms": int(row.start_ms),
            "gt_end_ms": int(row.end_ms), "diagnosis_status": "boundary_gt_excluded",
        } for row in boundary_events.itertuples(index=False)])
        raw_matching = pd.concat([raw_matching, boundary_rows], ignore_index=True, sort=False)
    result = {
        "raw_gt_total_intersecting_test": int(len(intersecting)),
        "raw_gt_complete": int(len(complete)),
        "raw_gt_cross_boundary": int(boundary),
        "detected_episodes": int(len(episodes)),
        "matched": int((raw_matching["match_status"] == "matched").sum()),
        "misses": int((raw_matching["match_status"] == "miss").sum()),
        "false_alarms": int((raw_matching["match_status"] == "false_alarm").sum()),
        "matched_diagnosis": diagnoses,
        "diagnosis_at_1_all_raw_gt_numerator": diagnoses["correct"],
        "diagnosis_at_1_all_raw_gt_denominator": int(len(intersecting)),
        "diagnosis_at_1_all_raw_gt": diagnoses["correct"] / len(intersecting) if len(intersecting) else None,
        "onset_to_detector_seconds_mean": float(np.mean(detector_delays)) if detector_delays else None,
        "onset_to_rca_data_ready_seconds_mean": float(np.mean(data_ready_delays)) if data_ready_delays else None,
        "final_diagnosis_wall_time": None,
        "final_diagnosis_wall_time_status": "UNMEASURED",
    }
    return result, raw_matching


def evaluate_c1_oracle_a(*, locked_case_details: Sequence[Mapping[str, object]],
                         index, arm_b: ConditionalLogitFit,
                         test_interval_ms: tuple):
    """Post-lock oracle context: B weights on GT-anchor features, same scope."""
    start, end = (int(value) for value in test_interval_ms)
    rows = []
    for case in locked_case_details:
        anchor = int(case["gt_start_ms"])
        status = "complete"
        ranking = ""
        if anchor - 300000 < start or anchor + 300000 > end:
            status = "illegal_gt_context"
        else:
            try:
                features = extract_c1_anchor_features(
                    index, case_id=str(case["case_id"]), anchor_ms=anchor)
                ranking = json.dumps(rank_candidates(GAIA_SERVICES, arm_b.scores(features)), separators=(",", ":"))
            except ValueError:
                status = "ranking_failure"
        score = _case_scores(_ranking(ranking), str(case["gt_service"]))
        rows.append({"prediction_id": str(case["prediction_id"]),
                     "case_id": str(case["case_id"]), "gt_anchor_ms": anchor,
                     "status": status, "ranking_a": ranking, **score})
    frame = pd.DataFrame(rows)
    n = len(frame)
    if not n:
        raise ValueError("C1 oracle A requires the locked matched-case scope")
    summary = {"n_same_as_bc": n, "complete": int((frame["status"] == "complete").sum()),
               "illegal_gt_context": int((frame["status"] == "illegal_gt_context").sum()),
               "ranking_failure": int((frame["status"] == "ranking_failure").sum()),
               "uses_b_weights": True,
               "diagnostic_only": True,
               "AC@1": float(frame["AC@1"].sum() / n),
               "AC@3": float(frame["AC@3"].sum() / n),
               "AC@5": float(frame["AC@5"].sum() / n),
               "MRR": float(frame["MRR"].sum() / n)}
    return frame, summary
