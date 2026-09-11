"""Metrics and baselines for the frozen P5-I1 two-stage evaluation."""

from __future__ import annotations

import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .protocol import GAIA_SERVICES


def _complete_ranking(value: Sequence[str]) -> Tuple[str, ...]:
    ranking = tuple(str(item) for item in value)
    if len(ranking) != len(GAIA_SERVICES) or set(ranking) != set(GAIA_SERVICES):
        raise ValueError("ranking must contain every canonical GAIA service exactly once")
    return ranking


def parse_ranking_row(row: Mapping[str, object]) -> Tuple[str, ...]:
    """Read a ranking from rank_1..rank_10 or ranking_json columns."""

    columns = tuple(str(row.get("rank_{}".format(index))) for index in range(1, 11))
    if set(columns) == set(GAIA_SERVICES):
        return _complete_ranking(columns)
    if "ranking_json" not in row:
        raise ValueError("prediction row lacks a complete service ranking")
    return _complete_ranking(json.loads(str(row["ranking_json"])))


def detector_only_rankings(
    matched: pd.DataFrame,
    node_predictions: pd.DataFrame,
) -> Mapping[str, Mapping[str, object]]:
    """Rank services by Ada-MGAD anomaly score at each matched ``t_hat``.

    Equal scores retain the fixed service registry order, as required by the
    detector-only baseline protocol.
    """

    required_matching = {"prediction_id", "t_hat", "match_status", "case_id", "gt_service"}
    required_scores = {"prediction_timestamp", "service", "anomaly_score"}
    if not required_matching.issubset(matched.columns):
        raise ValueError("matching table lacks detector-only identity columns")
    if not required_scores.issubset(node_predictions.columns):
        raise ValueError("node prediction table lacks timestamped service scores")
    selected = matched.loc[matched["match_status"].astype(str) == "matched"].copy()
    scores = node_predictions.copy()
    if "split" in scores.columns:
        scores = scores.loc[scores["split"].astype(str) == "test"]
    registry_index = {service: index for index, service in enumerate(GAIA_SERVICES)}
    output: Dict[str, Mapping[str, object]] = {}
    for row in selected.itertuples(index=False):
        anchor = int(row.t_hat)
        observed = scores.loc[scores["prediction_timestamp"].astype(np.int64) == anchor]
        if len(observed) != len(GAIA_SERVICES):
            raise ValueError("t_hat {} does not have exactly ten service scores".format(anchor))
        if observed["service"].duplicated().any() or set(observed["service"].astype(str)) != set(GAIA_SERVICES):
            raise ValueError("t_hat {} service scores are incomplete or duplicated".format(anchor))
        by_service = {
            str(item.service): float(item.anomaly_score)
            for item in observed.itertuples(index=False)
        }
        if not np.all(np.isfinite(tuple(by_service.values()))):
            raise ValueError("detector-only baseline contains a non-finite score")
        ranking = tuple(sorted(
            GAIA_SERVICES,
            key=lambda service: (-by_service[service], registry_index[service]),
        ))
        prediction_id = str(row.prediction_id)
        if prediction_id in output:
            raise ValueError("duplicate matched prediction ID")
        output[prediction_id] = {
            "prediction_id": prediction_id,
            "case_id": str(row.case_id),
            "t_hat": anchor,
            "gt_service": str(row.gt_service),
            "ranking": ranking,
            "scores": by_service,
        }
    return output


def diagnosis_metrics(
    matching: pd.DataFrame,
    rankings_by_prediction: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Compute the explicitly penalized full-pipeline Diagnosis@k metrics.

    For each k, an unmatched prediction is FP; an unmatched GT is FN; and a
    matched event whose labelled service is outside Top-k is both FP and FN.
    """

    if "match_status" not in matching or "prediction_id" not in matching:
        raise ValueError("matching table lacks match status/identity")
    matched = matching.loc[matching["match_status"].astype(str) == "matched"]
    false_alarms = int((matching["match_status"].astype(str) == "false_alarm").sum())
    misses = int((matching["match_status"].astype(str) == "miss").sum())
    matched_ids = set(matched["prediction_id"].astype(str))
    if set(str(key) for key in rankings_by_prediction) != matched_ids:
        raise ValueError("detected rankings must align exactly with matched predictions")
    result = {}
    for k in (1, 3, 5):
        successful = 0
        ranking_failures = 0
        for row in matched.itertuples(index=False):
            ranking = _complete_ranking(rankings_by_prediction[str(row.prediction_id)])
            if str(row.gt_service) in ranking[:k]:
                successful += 1
            else:
                ranking_failures += 1
        tp = successful
        fp = false_alarms + ranking_failures
        fn = misses + ranking_failures
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["@{}".format(k)] = {
            "diagnosis_true_positive": int(tp),
            "diagnosis_false_positive": int(fp),
            "diagnosis_false_negative": int(fn),
            "ranking_failures": int(ranking_failures),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "definition": {
            "TP@k": "matched prediction and labelled fault service in Ada-RCA-G Top-k",
            "FP@k": "unmatched prediction or matched prediction with labelled service outside Top-k",
            "FN@k": "unmatched GT or matched GT with labelled service outside Top-k",
            "note": "a matched ranking failure contributes to both FP and FN",
        },
        "counts": {
            "matched_events": int(len(matched)),
            "false_alarm_predictions": false_alarms,
            "missed_gt_events": misses,
            "predicted_episodes": int(len(matched) + false_alarms),
            "ground_truth_events": int(len(matched) + misses),
        },
        "metrics": result,
        "primary_summary": "Diagnosis F1@1",
        "diagnosis_f1_at_1": float(result["@1"]["f1"]),
    }


def anchor_metric_delta(
    oracle_metrics: Mapping[str, object],
    detected_metrics: Mapping[str, object],
) -> Mapping[str, float]:
    """Return detected-minus-oracle degradation on matched cases."""

    oracle = oracle_metrics["overall"]
    detected = detected_metrics["overall"]
    return {
        "delta_detected_minus_oracle_{}".format(metric):
            float(detected[metric]) - float(oracle[metric])
        for metric in ("AC@1", "AC@3", "AC@5", "MRR")
    }


def _correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool) -> object:
    if len(left) < 2 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return None
    if ranked:
        left = rankdata(left, method="average")
        right = rankdata(right, method="average")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def delay_ranking_relationship(
    matched: pd.DataFrame,
    rankings_by_prediction: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Describe signed/absolute detection delay versus ranking quality."""

    selected = matched.loc[matched["match_status"].astype(str) == "matched"].copy()
    records = []
    for row in selected.itertuples(index=False):
        ranking = _complete_ranking(rankings_by_prediction[str(row.prediction_id)])
        rank = ranking.index(str(row.gt_service)) + 1
        records.append({
            "delay": float(row.detection_delay_seconds),
            "absolute_delay": abs(float(row.detection_delay_seconds)),
            "reciprocal_rank": 1.0 / float(rank),
            "ac_at_1": float(rank == 1),
        })
    if not records:
        return {
            "matched_cases": 0,
            "signed_delay_vs_reciprocal_rank_pearson": None,
            "signed_delay_vs_reciprocal_rank_spearman": None,
            "absolute_delay_vs_reciprocal_rank_pearson": None,
            "absolute_delay_vs_reciprocal_rank_spearman": None,
            "delay_bands": {},
        }
    frame = pd.DataFrame(records)
    signed = frame["delay"].to_numpy(dtype=float)
    absolute = frame["absolute_delay"].to_numpy(dtype=float)
    reciprocal = frame["reciprocal_rank"].to_numpy(dtype=float)
    band_masks = {
        "early_lt_-30s": signed < -30.0,
        "near_-30s_to_30s": (signed >= -30.0) & (signed <= 30.0),
        "late_gt_30s": signed > 30.0,
    }
    bands = {}
    for name, mask in band_masks.items():
        subset = frame.loc[mask]
        bands[name] = {
            "case_count": int(len(subset)),
            "mean_detection_delay_seconds": (
                float(subset["delay"].mean()) if len(subset) else None
            ),
            "AC@1": float(subset["ac_at_1"].mean()) if len(subset) else None,
            "MRR": float(subset["reciprocal_rank"].mean()) if len(subset) else None,
        }
    return {
        "matched_cases": int(len(frame)),
        "signed_delay_vs_reciprocal_rank_pearson": _correlation(signed, reciprocal, ranked=False),
        "signed_delay_vs_reciprocal_rank_spearman": _correlation(signed, reciprocal, ranked=True),
        "absolute_delay_vs_reciprocal_rank_pearson": _correlation(absolute, reciprocal, ranked=False),
        "absolute_delay_vs_reciprocal_rank_spearman": _correlation(absolute, reciprocal, ranked=True),
        "delay_bands": bands,
    }
