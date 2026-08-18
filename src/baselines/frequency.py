"""Train-fold-only root-frequency sanity baseline."""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Tuple

from src.data.schema import RCACaseInput, RCACaseLabel, assert_label_free


@dataclass(frozen=True)
class RootFrequencyModel:
    """Immutable root counts learned from one training partition."""

    root_counts: Tuple[Tuple[str, int], ...]
    training_case_count: int

    def rank(self, case_input: RCACaseInput) -> Tuple[str, ...]:
        assert_label_free(case_input)
        counts = dict(self.root_counts)
        return tuple(
            sorted(
                case_input.services,
                key=lambda service: (-counts.get(service, 0), service),
            )
        )


def fit_root_frequency(labels: Sequence[RCACaseLabel]) -> RootFrequencyModel:
    """Fit counts using only labels explicitly supplied by the caller."""

    if not labels:
        raise ValueError("at least one training label is required")
    case_ids = [label.case_id for label in labels]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("training labels contain duplicate case_id")
    counts = Counter(label.root_service for label in labels)
    return RootFrequencyModel(
        root_counts=tuple(sorted(counts.items())),
        training_case_count=len(labels),
    )
