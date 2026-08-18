"""Deterministic per-case random ranking baseline."""

import hashlib
from typing import Tuple

from src.data.schema import RCACaseInput, assert_label_free


def deterministic_random_ranking(
    case_input: RCACaseInput, seed: int
) -> Tuple[str, ...]:
    """Return a stable pseudo-random permutation without global RNG state."""

    assert_label_free(case_input)

    def key(service: str):
        payload = "p1-random-v1\x00{}\x00{}\x00{}".format(
            int(seed), case_input.case_id, service
        )
        return hashlib.sha256(payload.encode("utf-8")).digest(), service

    return tuple(sorted(case_input.services, key=key))
