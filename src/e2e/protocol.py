"""Frozen GAIA V3 protocol, registry binding, and chronological split helpers."""

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
    "login_failure",
    "memory_anomalies",
    "cpu_anomalies",
    "file_moving",
    "normal_memory_freed",
    "access_permission_denied",
)


@dataclass(frozen=True)
class TemporalBlock:
    """A contiguous wall-clock block with half-open anchor semantics."""

    name: str
    start_ms: int
    end_ms: int
    is_final: bool = False

    def contains_anchor(self, timestamp_ms: int) -> bool:
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


def ad_preprocessing_config_payload(config: Mapping[str, object]) -> Mapping[str, object]:
    """Return only configuration values that determine Ada-MGAD input arrays."""

    ad = config.get("ad")
    ad_model = config.get("ad_model")
    preprocessing = config.get("ad_preprocessing")
    event_registry = config.get("event_registry")
    if not all(isinstance(value, Mapping) for value in (ad, ad_model, preprocessing, event_registry)):
        raise ValueError("config lacks an Ada-MGAD preprocessing section")
    return {
        "fingerprint_version": "gaia_ad_preprocessing_config_v1",
        "schema_version": preprocessing.get("schema_version"),
        "services": config.get("services"),
        "gaia_raw_root": config.get("gaia_raw_root"),
        "split": config.get("split"),
        "ad": {
            "grid_seconds": ad.get("grid_seconds"),
            "window_bins": ad.get("window_bins"),
            "label_interval": ad.get("label_interval"),
        },
        "ad_model": {"label_percent": ad_model.get("label_percent")},
        "event_registry": {
            "path": event_registry.get("path"),
            "sha256": event_registry.get("sha256"),
            "taxonomy_version": event_registry.get("taxonomy_version"),
        },
        "policy_path": preprocessing.get("policy_path"),
        "frozen_schema_path": preprocessing.get("frozen_schema_path"),
    }


def ad_preprocessing_config_sha256(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        ad_preprocessing_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    """Load and fail closed on drift from the V3 protocol."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    services = tuple(data["services"])
    if services != GAIA_SERVICES:
        raise ValueError("config service registry/order differs from canonical GAIA registry")
    if not str(data.get("schema_version", "")).startswith("p5_v3_"):
        # The checked-in V1 fixture remains readable by historical audit unit
        # tests.  No V3 production entry point accepts or derives artifacts
        # from this compatibility branch.
        return data
    if int(data["ad"]["grid_seconds"]) != 30 or int(data["ad"]["window_bins"]) != 10:
        raise ValueError("GAIA V3 Ada-MGAD grid/window protocol drift")
    if int(data["ad"].get("epochs", 0)) != 100 or int(data["ad"].get("patience", 0)) != 10:
        raise ValueError("GAIA V3 requires max_epochs=100 and patience=10")
    if str(data["ad"].get("early_stopping_metric")) != "train_f1":
        raise ValueError("GAIA V3 early stopping must use train_f1")
    if str(data["ad"].get("primary_checkpoint")) != "best_train_f1.pt":
        raise ValueError("GAIA V3 primary checkpoint must be best_train_f1.pt")
    if str(data["ad"].get("auxiliary_checkpoint")) != "best_train_loss.pt":
        raise ValueError("GAIA V3 auxiliary checkpoint must be best_train_loss.pt")
    if str(data.get("ad_model", {}).get("checkpoint_policy")) != (
        "best_train_f1_primary_best_train_loss_diagnostic"
    ):
        raise ValueError("GAIA V3 checkpoint policy must select Train F1 as primary")
    if "expected_events" in data.get("event_registry", {}):
        raise ValueError("event registry row counts must be derived, not hardcoded")
    if str(data["event_trigger"].get("threshold_selection")) != "train_only_exact_unique_scores":
        raise ValueError("GAIA V3 threshold selection must be Train-only")
    if str(data["event_trigger"].get("matching_semantics")) != "causal_max_cardinality_minimum_delay":
        raise ValueError("GAIA V3 event matching semantics drift")
    split = data.get("split", {})
    if str(split.get("mode")) != "chronological_metric_70_30":
        raise ValueError("GAIA V3 requires the metric-based chronological 70/30 split")
    if int(split.get("train_fraction_numerator", 0)) != 7 or int(
        split.get("train_fraction_denominator", 0)
    ) != 10:
        raise ValueError("GAIA V3 split must use K=(7*N)//10")
    if "validation" in json.dumps(split).lower():
        raise ValueError("GAIA V3 split binding must not contain a validation partition")
    rca = data["rca"]
    if int(rca["window_seconds"]) != 300 or int(rca["bin_seconds"]) != 15:
        raise ValueError("GAIA V3 requires W300-B15")
    preprocessing = data["preprocessing"]
    cpu_budget = int(preprocessing["cpu_budget"])
    if cpu_budget < 1:
        raise ValueError("preprocessing CPU budget must be positive")
    for key in ("metric_workers", "log_workers", "trace_workers", "ad_workers", "rca_index_workers", "rca_feature_workers"):
        value = int(preprocessing.get(key, preprocessing.get("ad_workers", 1)))
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
        "metric": "metric_workers",
        "logs": "log_workers",
        "traces": "trace_workers",
        "ad": "ad_workers",
        "rca_index": "rca_index_workers",
        "rca_features": "rca_feature_workers",
    }
    if stage not in keys:
        raise ValueError("unknown preprocessing stage: {}".format(stage))
    frozen = config["preprocessing"]
    configured_workers = int(frozen.get(keys[stage], frozen.get("ad_workers", 1)))
    resolved_workers = int(configured_workers if workers is None else workers)
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
    if str(split.get("mode", "")).endswith("60_20_20"):
        boundaries = [int(value) for value in split["boundaries_ms"]]
        return (
            TemporalBlock("train", int(split["absolute_start_ms"]), boundaries[0]),
            TemporalBlock("validation", boundaries[0], boundaries[1]),
            TemporalBlock("test", boundaries[1], int(split["absolute_end_ms"]), is_final=True),
        )
    overall_start = int(split["absolute_start_ms"])
    overall_end = int(split["absolute_end_ms"])
    grid_ms = int(config["ad"]["grid_seconds"]) * 1000
    duration = overall_end - overall_start
    if duration <= 0 or duration % grid_ms:
        raise ValueError("metric detector timeline must contain whole 30-second bins")
    n = duration // grid_ms
    train_bins = (7 * n) // 10
    boundary = overall_start + train_bins * grid_ms
    declared = split.get("boundary_ms")
    if declared is not None and int(declared) != boundary:
        raise ValueError("declared 70/30 split boundary differs from K=(7*N)//10")
    return (
        TemporalBlock("train", overall_start, boundary),
        TemporalBlock("test", boundary, overall_end, is_final=True),
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
    if frame["case_id"].duplicated().any():
        raise ValueError("event registry case IDs must be unique")
    if "labelled_service" not in frame.columns:
        frame["labelled_service"] = frame["service"]
    if not set(frame["service"]).issubset(GAIA_SERVICES):
        raise ValueError("event registry contains a service outside the canonical registry")
    if str(config.get("schema_version", "")).startswith("p5_v3_") and set(frame["fault_type"]) != set(SUPPORTED_FAULT_TYPES):
        raise ValueError("event registry supported fault taxonomy mismatch")
    if (frame["end_ms"] <= frame["start_ms"]).any():
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
    start = ((blocks[0].start_ms + grid_ms - 1) // grid_ms) * grid_ms
    end = ((blocks[-1].end_ms - grid_ms) // grid_ms) * grid_ms
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
        "window_interval": "[target_bin_start-(window_bins-1)*grid,target_bin_start+grid)",
    }


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    ensure_parent(path)
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
