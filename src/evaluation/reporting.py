"""Overall and label-stratified reporting for complete RCA rankings."""

from typing import Dict, Mapping, Sequence

from src.data.schema import RCACaseInput, RCACaseLabel
from src.evaluation.evaluator import EvaluationSummary, evaluate_rankings


def _metric_record(summary: EvaluationSummary) -> Dict[str, float]:
    return {
        "AC@1": summary.ac_at[1],
        "AC@3": summary.ac_at[3],
        "AC@5": summary.ac_at[5],
        "Avg@5": summary.avg_at_5,
        "MRR": summary.mrr,
    }


def _stratified_metrics(
    inputs: Sequence[RCACaseInput],
    labels: Sequence[RCACaseLabel],
    rankings: Mapping[str, Sequence[str]],
    field_name: str,
) -> Mapping[str, object]:
    inputs_by_id = {row.case_id: row for row in inputs}
    case_ids_by_category = {}
    for label in labels:
        value = getattr(label, field_name)
        category = "<none>" if value is None else str(value)
        case_ids_by_category.setdefault(category, []).append(label.case_id)

    by_category = {}
    for category, case_ids in sorted(case_ids_by_category.items()):
        wanted = set(case_ids)
        category_inputs = tuple(inputs_by_id[case_id] for case_id in case_ids)
        category_labels = tuple(label for label in labels if label.case_id in wanted)
        category_rankings = {case_id: rankings[case_id] for case_id in case_ids}
        summary = evaluate_rankings(
            category_inputs, category_labels, category_rankings
        )
        by_category[category] = {
            "case_count": len(case_ids),
            **_metric_record(summary),
        }

    metric_names = ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
    macro = {
        metric: sum(row[metric] for row in by_category.values()) / len(by_category)
        for metric in metric_names
    }
    return {"by_category": by_category, "category_count": len(by_category), "macro": macro}


def evaluate_ranking_report(
    inputs: Sequence[RCACaseInput],
    labels: Sequence[RCACaseLabel],
    rankings: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Report overall plus fault-type and root-service macro metrics."""

    overall = evaluate_rankings(inputs, labels, rankings)
    return {
        "fault_type": _stratified_metrics(
            inputs, labels, rankings, "fault_type"
        ),
        "overall": {"case_count": overall.total_cases, **_metric_record(overall)},
        "root_service": _stratified_metrics(
            inputs, labels, rankings, "root_service"
        ),
    }
