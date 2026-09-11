"""Frozen P5-I1 protocol, registry binding, and chronological split helpers."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import pandas as pd


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
SUPPORTED_FAULT_TYPES = (
    "login failure",
    "memory_anomalies",
    "cpu_anomalies",
    "file moving program",
    "access permission denied exception",
)


@dataclass(frozen=True)
class TemporalBlock:
    """A contiguous wall-clock block with half-open anchor semantics."""

    name: str
    start_ms: int
    end_ms: int
    is_final: bool = False

    def contains_anchor(self, timestamp_ms: int) -> bool:
        if self.is_final:
            return self.start_ms <= timestamp_ms <= self.end_ms
        return self.start_ms <= timestamp_ms < self.end_ms

    def contains_interval(self, start_ms: int, end_ms: int) -> bool:
        # Injection intervals are complete physical intervals, so their end may
        # coincide with a split boundary without crossing it.
        return self.start_ms <= start_ms and end_ms <= self.end_ms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_digest(root: Path, paths: Iterable[Path]) -> Mapping[str, object]:
    """Bind a telemetry mirror by relative paths and byte sizes without reading it."""

    root = Path(root)
    records = sorted(
        ((str(Path(path).relative_to(root)), Path(path).stat().st_size) for path in paths),
        key=lambda item: item[0],
    )
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


def load_config(path: Path) -> Mapping[str, object]:
    """Load the JSON-compatible YAML used to avoid an optional YAML runtime."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    services = tuple(data["services"])
    if services != GAIA_SERVICES:
        raise ValueError("config service registry/order differs from canonical GAIA registry")
    if int(data["ad"]["grid_seconds"]) != 30 or int(data["ad"]["window_bins"]) != 10:
        raise ValueError("P5-I1 Ada-MGAD grid/window protocol drift")
    rca = data["rca"]
    if int(rca["window_seconds"]) != 300 or int(rca["bin_seconds"]) != 15:
        raise ValueError("P5-I1 requires W300-B15")
    preprocessing = data["preprocessing"]
    cpu_budget = int(preprocessing["cpu_budget"])
    if cpu_budget < 1:
        raise ValueError("preprocessing CPU budget must be positive")
    for key in ("ad_workers", "rca_index_workers", "rca_feature_workers"):
        value = int(preprocessing[key])
        if value < 1 or value > cpu_budget:
            raise ValueError("{} must be within the preprocessing CPU budget".format(key))
    if str(preprocessing["multiprocessing_start_method"]) not in ("spawn", "forkserver"):
        raise ValueError("unsupported preprocessing multiprocessing start method")
    return data


def preprocessing_runtime(
    config: Mapping[str, object],
    stage: str,
    *,
    workers: int = None,
    chunk_rows: int = None,
    case_chunk_size: int = None,
    start_method: str = None,
) -> Mapping[str, object]:
    """Resolve bounded runtime controls without changing scientific protocol."""

    keys = {
        "ad": "ad_workers",
        "rca_index": "rca_index_workers",
        "rca_features": "rca_feature_workers",
    }
    if stage not in keys:
        raise ValueError("unknown preprocessing stage: {}".format(stage))
    frozen = config["preprocessing"]
    resolved_workers = int(
        frozen[keys[stage]] if workers is None else workers
    )
    cpu_budget = int(frozen["cpu_budget"])
    if resolved_workers < 1 or resolved_workers > cpu_budget:
        raise ValueError(
            "preprocessing workers must be between 1 and configured CPU budget {}".format(
                cpu_budget
            )
        )
    resolved_chunk_rows = int(
        frozen["chunk_rows"] if chunk_rows is None else chunk_rows
    )
    if resolved_chunk_rows < 1:
        raise ValueError("preprocessing chunk rows must be positive")
    resolved_start = str(
        frozen["multiprocessing_start_method"] if start_method is None else start_method
    )
    if resolved_start not in ("spawn", "forkserver"):
        raise ValueError("unsupported preprocessing multiprocessing start method")
    resolved_case_chunk = int(
        frozen["rca_case_chunk_size"] if case_chunk_size is None else case_chunk_size
    )
    if resolved_case_chunk < 1:
        raise ValueError("RCA case chunk size must be positive")
    return {
        "workers": resolved_workers,
        "chunk_rows": resolved_chunk_rows,
        "case_chunk_size": resolved_case_chunk,
        "start_method": resolved_start,
        "cpu_budget": cpu_budget,
        "memory_budget_gb": int(frozen["memory_budget_gb"]),
    }


def temporal_blocks(config: Mapping[str, object]) -> Tuple[TemporalBlock, ...]:
    split = config["split"]
    boundaries = tuple(int(value) for value in split["boundaries_ms"])
    overall_start = int(split["absolute_start_ms"])
    overall_end = int(split["absolute_end_ms"])
    if not overall_start < boundaries[0] < boundaries[1] < overall_end:
        raise ValueError("chronological split boundaries must be strictly ordered")
    return (
        TemporalBlock("train", overall_start, boundaries[0]),
        TemporalBlock("validation", boundaries[0], boundaries[1]),
        TemporalBlock("test", boundaries[1], overall_end, is_final=True),
    )


def load_registry(config: Mapping[str, object], project_root: Path) -> pd.DataFrame:
    binding = config["event_registry"]
    path = (Path(project_root) / str(binding["path"])).resolve()
    actual_hash = sha256_file(path)
    if actual_hash != str(binding["sha256"]):
        raise ValueError("event registry SHA-256 mismatch")
    frame = pd.read_csv(path)
    required = {
        "case_id", "source_index", "service", "fault_type", "start_ms", "end_ms"
    }
    if not required.issubset(frame.columns):
        raise ValueError("event registry is missing required columns")
    if len(frame) != int(binding["expected_events"]):
        raise ValueError("event registry row count mismatch")
    if frame["case_id"].duplicated().any():
        raise ValueError("event registry case IDs must be unique")
    if not set(frame["service"]).issubset(GAIA_SERVICES):
        raise ValueError("event registry contains a service outside the canonical registry")
    if set(frame["fault_type"]) != set(SUPPORTED_FAULT_TYPES):
        raise ValueError("event registry supported fault taxonomy mismatch")
    if (frame["end_ms"] < frame["start_ms"]).any():
        raise ValueError("event registry contains a negative interval")
    return frame.sort_values(["start_ms", "source_index", "case_id"], kind="stable").reset_index(drop=True)


def assign_event_blocks(
    registry: pd.DataFrame, blocks: Sequence[TemporalBlock]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign complete raw injections to blocks; crossing injections are purged."""

    assigned = []
    purged = []
    for row in registry.itertuples(index=False):
        matches = [
            block.name
            for block in blocks
            if block.contains_interval(int(row.start_ms), int(row.end_ms))
        ]
        record = row._asdict()
        if len(matches) == 1:
            record["split"] = matches[0]
            assigned.append(record)
        else:
            record["purge_reason"] = "raw_injection_crosses_split_boundary"
            purged.append(record)
    return pd.DataFrame(assigned), pd.DataFrame(purged)


def purge_rca_cases(
    assigned: pd.DataFrame,
    blocks: Sequence[TemporalBlock],
    window_seconds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Purge cases whose exact ``[t0-W,t0+W)`` context crosses its block."""

    by_name = {block.name: block for block in blocks}
    first_name = blocks[0].name
    final_name = blocks[-1].name
    retained = []
    purged = []
    radius_ms = int(window_seconds) * 1000
    for row in assigned.itertuples(index=False):
        block = by_name[str(row.split)]
        context_start = int(row.start_ms) - radius_ms
        context_end = int(row.start_ms) + radius_ms
        record = row._asdict()
        record["context_start_ms"] = context_start
        record["context_end_ms"] = context_end
        crosses_left_split = block.name != first_name and context_start < block.start_ms
        crosses_right_split = block.name != final_name and context_end > block.end_ms
        if not crosses_left_split and not crosses_right_split:
            retained.append(record)
        else:
            record["purge_reason"] = "w{}_context_crosses_split_boundary".format(
                int(window_seconds)
            )
            purged.append(record)
    return pd.DataFrame(retained), pd.DataFrame(purged)


def distribution(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty:
        return {}
    counts = frame[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def count_crossing_ad_windows(
    blocks: Sequence[TemporalBlock], grid_seconds: int, window_bins: int
) -> Mapping[str, object]:
    """Count epoch-aligned prediction windows removed at split boundaries."""

    grid_ms = int(grid_seconds) * 1000
    lookback_ms = (int(window_bins) - 1) * grid_ms
    start = (blocks[0].start_ms // grid_ms) * grid_ms
    end = (blocks[-1].end_ms // grid_ms) * grid_ms
    crossing = []
    by_split = {block.name: 0 for block in blocks}
    outside_absolute_range = 0
    total = 0
    timestamp = start
    while timestamp <= end:
        total += 1
        left = timestamp - lookback_ms
        right = timestamp + grid_ms
        owner = next((block for block in blocks if block.contains_anchor(timestamp)), None)
        if owner is None or not owner.contains_interval(left, right):
            crossing.append(timestamp)
            if owner is None:
                outside_absolute_range += 1
            else:
                by_split[owner.name] += 1
        timestamp += grid_ms
    return {
        "candidate_prediction_timestamps": total,
        "purged_windows": len(crossing),
        "purged_windows_by_split": by_split,
        "unassigned_grid_timestamps_outside_absolute_range": outside_absolute_range,
        "purged_prediction_timestamps_ms": crossing,
        "window_interval": "[prediction_timestamp-(window_bins-1)*grid, prediction_timestamp+grid)",
    }


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    ensure_parent(path)
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
