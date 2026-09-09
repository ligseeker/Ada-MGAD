"""Shared deterministic helpers for the P5-G0R2 audit scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


GAIA_SERVICES = (
    "dbservice1",
    "dbservice2",
    "logservice1",
    "logservice2",
    "mobservice1",
    "mobservice2",
    "redisservice1",
    "redisservice2",
    "webservice1",
    "webservice2",
)

SUPPORTED_FAULTS = frozenset(
    {
        "access permission denied exception",
        "cpu_anomalies",
        "file moving program",
        "login failure",
        "memory_anomalies",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_digest(root: Path, paths: Iterable[Path]) -> Mapping[str, object]:
    """Match the historical size-bound inventory digest without reading 30 GB."""

    records = []
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value.relative_to(root))):
        relative = str(path.relative_to(root))
        records.append((relative, path.stat().st_size))
    digest = hashlib.sha256()
    for relative, size in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return {
        "files": len(records),
        "bytes": sum(size for _, size in records),
        "layout_sha256": digest.hexdigest(),
        "digest_scope": "sha256(sorted relative_path + NUL + byte_size)",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def percentile_summary(values: Sequence[float]) -> Mapping[str, object]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.percentile(array, 50)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def opaque_case_id(source_index: int, start_ms: int) -> str:
    payload = "gaia-2021-07:{}:{}".format(source_index, start_ms).encode("utf-8")
    return "gaia-{}".format(hashlib.sha256(payload).hexdigest()[:16])
