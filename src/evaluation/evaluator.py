"""Dataset-level evaluator with complete-ranking and Label Firewall checks."""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

from src.data.schema import (
    RCACaseInput,
    RCACaseLabel,
    SchemaValidationError,
    assert_label_free,
    validate_case_collection,
)
from src.evaluation.metrics import average_at_k, hit_at_k, reciprocal_rank


Ranking = Tuple[str, ...]


def validate_ranking(case_input: RCACaseInput, ranking: Sequence[str]) -> Ranking:
    """Require a duplicate-free permutation of the legal candidate services."""

    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise SchemaValidationError("ranking must be a sequence of service names")
    normalized = tuple(ranking)
    if any(not isinstance(service, str) for service in normalized):
        raise SchemaValidationError("ranking entries must be service names")
    if len(set(normalized)) != len(normalized):
        raise SchemaValidationError(
            "ranking for case {!r} contains duplicate services".format(case_input.case_id)
        )

    expected = set(case_input.services)
    observed = set(normalized)
    if len(normalized) != len(case_input.services) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SchemaValidationError(
            "ranking for case {!r} must cover exactly services; missing={}, extra={}".format(
                case_input.case_id, missing, extra
            )
        )
    return normalized


def predict_rankings(
    inputs: Sequence[RCACaseInput],
    predictor: Callable[[RCACaseInput], Sequence[str]],
) -> Dict[str, Ranking]:
    """Run a predictor through an input-only boundary.

    Labels are intentionally absent from this function's signature. This is the
    minimal executable Label Firewall for P1 predictors and baselines.
    """

    validate_case_collection(inputs)
    predictions = {}
    for case_input in inputs:
        assert_label_free(case_input)
        predictions[case_input.case_id] = validate_ranking(
            case_input, predictor(case_input)
        )
    return predictions


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    root_service: str
    root_rank: int
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    ac_at: Mapping[int, float]
    avg_at_5: float
    mrr: float
    per_case: Tuple[CaseEvaluation, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "AC@1": self.ac_at[1],
            "AC@3": self.ac_at[3],
            "AC@5": self.ac_at[5],
            "Avg@5": self.avg_at_5,
            "MRR": self.mrr,
            "per_case": [
                {
                    "case_id": row.case_id,
                    "root_service": row.root_service,
                    "root_rank": row.root_rank,
                    "reciprocal_rank": row.reciprocal_rank,
                }
                for row in self.per_case
            ],
        }


def evaluate_rankings(
    inputs: Sequence[RCACaseInput],
    labels: Sequence[RCACaseLabel],
    rankings: Mapping[str, Sequence[str]],
) -> EvaluationSummary:
    """Evaluate complete service rankings for single-root RCA cases."""

    if not inputs:
        raise SchemaValidationError("at least one RCA case is required")
    validate_case_collection(inputs, labels)

    input_ids = {case.case_id for case in inputs}
    ranking_ids = set(rankings)
    if input_ids != ranking_ids:
        missing = sorted(input_ids - ranking_ids)
        extra = sorted(ranking_ids - input_ids)
        raise SchemaValidationError(
            "input/ranking case_id mismatch; missing rankings={}, extra rankings={}".format(
                missing, extra
            )
        )

    labels_by_id = {label.case_id: label for label in labels}
    hits = {k: [] for k in range(1, 6)}
    avg_at_5_values = []
    reciprocal_ranks = []
    per_case = []

    for case_input in inputs:
        ranking = validate_ranking(case_input, rankings[case_input.case_id])
        label = labels_by_id[case_input.case_id]
        root_rank = ranking.index(label.root_service) + 1
        rr = reciprocal_rank(ranking, label.root_service)
        for k in range(1, 6):
            hits[k].append(hit_at_k(ranking, label.root_service, k))
        avg_at_5_values.append(average_at_k(ranking, label.root_service, 5))
        reciprocal_ranks.append(rr)
        per_case.append(
            CaseEvaluation(
                case_id=case_input.case_id,
                root_service=label.root_service,
                root_rank=root_rank,
                reciprocal_rank=rr,
            )
        )

    count = len(inputs)
    return EvaluationSummary(
        total_cases=count,
        ac_at={k: sum(hits[k]) / count for k in (1, 3, 5)},
        avg_at_5=sum(avg_at_5_values) / count,
        mrr=sum(reciprocal_ranks) / count,
        per_case=tuple(per_case),
    )
