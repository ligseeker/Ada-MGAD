"""Validation and immutable loading for GAIA preprocessing schemas."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCHEMA_VERSION = "gaia_ad_preprocessing_v2"
LOG_LEVELS = ("INFO", "WARNING", "ERROR", "DEBUG", "UNKNOWN")
TRACE_STATUSES = ("200", "300", "400", "500")
TRACE_STATISTICS = ("count", "mean_latency")
TRACE_SLOTS = tuple(
    "{}_{}_log".format(status, statistic)
    for status in TRACE_STATUSES
    for statistic in TRACE_STATISTICS
)
METRIC_OBSERVABILITY_SLOTS = (
    "global_observed_fraction",
    "host_applicable",
    "host_observed_fraction",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenPreprocessingSchema:
    """A validated schema whose payload cannot be mutated through this object."""

    path: Path
    sha256: str
    _canonical_json: str

    @property
    def payload(self) -> Dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def metric_slots(self) -> List[str]:
        return list(self.payload["metric"]["ordered_slots"])

    @property
    def log_slots(self) -> List[str]:
        return list(self.payload["logs"]["ordered_slots"])

    @property
    def trace_slots(self) -> List[str]:
        return list(self.payload["traces"]["ordered_slots"])

    @property
    def dimensions(self) -> Mapping[str, int]:
        return {
            "raw_node": len(self.metric_slots),
            "log_len": len(self.log_slots),
            "raw_edge": len(self.trace_slots),
        }


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(key))
    return value


def _require_unique_strings(value: Any, field: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("{} must be a list of non-empty strings".format(field))
    if len(value) != len(set(value)):
        raise ValueError("{} contains duplicate slots".format(field))
    return list(value)


def _reject_non_finite(value: Any, field: str = "schema") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("{} contains NaN or Inf".format(field))
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, "{}.{}".format(field, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_non_finite(child, "{}[{}]".format(field, index))


def _validate_source_binding(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_binding must be a non-empty object")
    required = {"config_sha256", "policy_sha256", "raw_train_sha256"}
    missing = required.difference(value)
    if missing:
        raise ValueError("source_binding is missing {}".format(sorted(missing)))
    for key, digest in value.items():
        if key.endswith("sha256") and (
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("source_binding.{} must be a lowercase SHA-256".format(key))


def _expected_log_slots(stable_cluster_ids: Sequence[Any]) -> List[str]:
    stable = ["stable_template_{}".format(cluster_id) for cluster_id in stable_cluster_ids]
    return (
        stable
        + ["RARE_{}".format(level) for level in LOG_LEVELS]
        + ["UNK_{}".format(level) for level in LOG_LEVELS]
        + ["level_{}".format(level) for level in LOG_LEVELS]
    )


def _validate(payload: Mapping[str, Any]) -> None:
    _reject_non_finite(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be {}".format(SCHEMA_VERSION))
    if payload.get("status") != "FROZEN":
        raise ValueError("status must be FROZEN")
    if payload.get("fit_split") != "train":
        raise ValueError("fit_split must be train")
    if payload.get("decision_inputs") != ["train"]:
        raise ValueError("decision_inputs must be exactly ['train']")
    if payload.get("gt_labels_used") is not False:
        raise ValueError("gt_labels_used must be false")
    if payload.get("test_used_for_selection") is not False:
        raise ValueError("test_used_for_selection must be false")
    _validate_source_binding(payload.get("source_binding"))

    metric = _require_mapping(payload, "metric")
    metric_slots = _require_unique_strings(metric.get("ordered_slots"), "metric.ordered_slots")
    if not 48 <= len(metric_slots) <= 64:
        raise ValueError("metric dimension must be within the frozen budget [48, 64]")
    if tuple(metric_slots[-len(METRIC_OBSERVABILITY_SLOTS) :]) != METRIC_OBSERVABILITY_SLOTS:
        raise ValueError("metric.ordered_slots must end with the three observability slots")

    logs = _require_mapping(payload, "logs")
    stable_cluster_ids = logs.get("stable_cluster_ids")
    if (
        not isinstance(stable_cluster_ids, list)
        or not stable_cluster_ids
        or not all(
            (isinstance(cluster_id, int) and not isinstance(cluster_id, bool))
            or (isinstance(cluster_id, str) and cluster_id)
            for cluster_id in stable_cluster_ids
        )
    ):
        raise ValueError("logs.stable_cluster_ids must contain non-empty integer/string IDs")
    if len(stable_cluster_ids) != len(set(stable_cluster_ids)):
        raise ValueError("logs.stable_cluster_ids contains duplicates")
    log_slots = _require_unique_strings(logs.get("ordered_slots"), "logs.ordered_slots")
    if log_slots != _expected_log_slots(stable_cluster_ids):
        raise ValueError("logs.ordered_slots does not match stable + RARE + UNK + level order")
    if not 32 <= len(log_slots) <= 64:
        raise ValueError("log dimension must be within the frozen budget [32, 64]")

    traces = _require_mapping(payload, "traces")
    if traces.get("status_order") != list(TRACE_STATUSES):
        raise ValueError("traces.status_order must be 200/300/400/500")
    if traces.get("statistic_order") != list(TRACE_STATISTICS):
        raise ValueError("traces.statistic_order must be count/mean_latency")
    trace_slots = _require_unique_strings(traces.get("ordered_slots"), "traces.ordered_slots")
    if trace_slots != list(TRACE_SLOTS):
        raise ValueError("traces.ordered_slots must be the fixed 8D count/mean_latency schema")

    edges = traces.get("directed_edges")
    if not isinstance(edges, list):
        raise ValueError("traces.directed_edges must be a list")
    normalized_edges = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(service, str) and service for service in edge)
            or edge[0] == edge[1]
        ):
            raise ValueError("each directed edge must contain two distinct service names")
        normalized_edges.append(tuple(edge))
    if len(normalized_edges) != len(set(normalized_edges)):
        raise ValueError("traces.directed_edges contains duplicates")


def load_frozen_preprocessing_schema(path: Path) -> FrozenPreprocessingSchema:
    """Load and validate a frozen schema without permitting implicit repairs."""

    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen preprocessing schema JSON: {}".format(exc)) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("frozen preprocessing schema must be a JSON object")
    _validate(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return FrozenPreprocessingSchema(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        _canonical_json=canonical,
    )


def verify_frozen_schema_sources(
    schema: FrozenPreprocessingSchema,
    *,
    config_path: Path,
    policy_path: Path,
    raw_train_path: Path,
) -> None:
    """Verify the three decision materials bound by a frozen schema."""

    if not isinstance(schema, FrozenPreprocessingSchema):
        raise TypeError("schema must be FrozenPreprocessingSchema")
    binding = schema.payload["source_binding"]
    paths = {
        "config_sha256": Path(config_path),
        "policy_sha256": Path(policy_path),
        "raw_train_sha256": Path(raw_train_path),
    }
    for key, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != binding[key]:
            raise ValueError("{} mismatch for {}".format(key, path.resolve()))
