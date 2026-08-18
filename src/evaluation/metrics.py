"""Pure functions for single-root RCA ranking metrics."""

from typing import Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _root_rank(ranking: Sequence[str], root_service: str) -> int:
    try:
        return ranking.index(root_service) + 1
    except ValueError:
        raise ValueError("root_service is absent from ranking")


def hit_at_k(ranking: Sequence[str], root_service: str, k: int) -> float:
    """Return 1.0 when the root occurs in the first ``k`` positions."""

    _validate_k(k)
    return float(_root_rank(ranking, root_service) <= k)


def average_at_k(ranking: Sequence[str], root_service: str, k: int) -> float:
    """Average AC@1 through AC@k for one case."""

    _validate_k(k)
    return sum(hit_at_k(ranking, root_service, j) for j in range(1, k + 1)) / k


def reciprocal_rank(ranking: Sequence[str], root_service: str) -> float:
    """Return reciprocal rank for one single-root case."""

    return 1.0 / _root_rank(ranking, root_service)
