"""Ranking validation and metrics for standalone RCA."""

from .bootstrap import paired_root_macro_bootstrap

from .evaluator import (
    CaseEvaluation,
    EvaluationSummary,
    evaluate_rankings,
    predict_rankings,
    validate_ranking,
)
from .metrics import average_at_k, hit_at_k, reciprocal_rank
from .reporting import evaluate_ranking_report

__all__ = [
    "CaseEvaluation",
    "EvaluationSummary",
    "average_at_k",
    "evaluate_rankings",
    "evaluate_ranking_report",
    "hit_at_k",
    "paired_root_macro_bootstrap",
    "predict_rankings",
    "reciprocal_rank",
    "validate_ranking",
]
