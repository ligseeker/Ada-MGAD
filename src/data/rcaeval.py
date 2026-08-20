"""Adapter for the local RCAEval failure-case layouts (RE2-OB, RE2-TT).

Both releases share one directory convention, one raw file set and one candidate
derivation rule, so a single reader serves them; only the dataset identity and the
auxiliary-entity filter are release specific. Those differences live in a frozen
``RCAEvalDatasetProfile`` rather than in forked code, because the RE2-OB manifest
under ``artifacts/p1/manifests/re2ob/`` is frozen: any behavioural drift in the
shared path would silently invalidate already-recorded case IDs and digests.
"""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import FrozenSet, List, Sequence, Tuple

from src.data.schema import (
    RCACaseInput,
    RCACaseLabel,
    TelemetryRef,
    validate_case_collection,
)


_AUXILIARY_METRIC_ENTITIES = frozenset(
    {
        "frontend-check",
        "frontend-external",
        "istio-init",
        "loadgenerator",
    }
)


@dataclass(frozen=True)
class RCAEvalDatasetProfile:
    """Release-specific identity for one RCAEval failure-case collection."""

    key: str
    dataset: str
    id_namespace: str
    id_prefix: str
    uri_namespace: str
    auxiliary_entities: FrozenSet[str]
    replicates: Tuple[str, ...]


RE2OB_PROFILE = RCAEvalDatasetProfile(
    key="re2ob",
    dataset="RCAEval-RE2-OB",
    id_namespace="RCAEval:RE2-OB",
    id_prefix="re2ob",
    uri_namespace="re2-ob",
    auxiliary_entities=_AUXILIARY_METRIC_ENTITIES,
    replicates=("1", "2", "3"),
)


RE2TT_PROFILE = RCAEvalDatasetProfile(
    key="re2tt",
    dataset="RCAEval-RE2-TT",
    id_namespace="RCAEval:RE2-TT",
    id_prefix="re2tt",
    uri_namespace="re2-tt",
    auxiliary_entities=_AUXILIARY_METRIC_ENTITIES,
    replicates=("1", "2", "3"),
)


DATASET_PROFILES = {
    RE2OB_PROFILE.key: RE2OB_PROFILE,
    RE2TT_PROFILE.key: RE2TT_PROFILE,
}


@dataclass(frozen=True)
class RCAEvalCaseSource:
    case_id: str
    relative_directory: str
    metrics_path: str
    logs_path: str
    traces_path: str
    inject_time_path: str


@dataclass(frozen=True)
class RCAEvalExcludedCase:
    relative_directory: str
    reason: str


@dataclass(frozen=True)
class RCAEvalAdapterResult:
    inputs: Tuple[RCACaseInput, ...]
    labels: Tuple[RCACaseLabel, ...]
    sources: Tuple[RCAEvalCaseSource, ...]
    excluded: Tuple[RCAEvalExcludedCase, ...]


def _opaque_case_id(profile: RCAEvalDatasetProfile, relative_directory: str) -> str:
    payload = "{}:{}".format(profile.id_namespace, relative_directory).encode("utf-8")
    return "{}-{}".format(profile.id_prefix, hashlib.sha256(payload).hexdigest()[:16])


def _parse_condition(condition: str) -> Tuple[str, str]:
    if "_" not in condition:
        raise ValueError("condition directory must be <root_service>_<fault_type>")
    root_service, fault_type = condition.rsplit("_", 1)
    return root_service, fault_type


def _read_header(path: Path) -> Sequence[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def _candidate_services(
    case_directory: Path, auxiliary_entities: FrozenSet[str]
) -> Tuple[str, ...]:
    simple_metrics = case_directory / "simple_metrics.csv"
    if simple_metrics.is_file():
        header = _read_header(simple_metrics)
        candidates = {
            column.rsplit("_", 1)[0]
            for column in header
            if column.endswith("_cpu") or column.endswith("_mem")
        }
        candidates.difference_update(auxiliary_entities)
    else:
        header = _read_header(case_directory / "metrics.csv")
        candidates = {
            column.split("_container-", 1)[0]
            for column in header
            if "_container-" in column
            and not column.startswith("loadgenerator_container-")
        }
        candidates.difference_update(auxiliary_entities)
    if not candidates:
        raise ValueError("could not derive observable service candidates")
    return tuple(sorted(candidates))


def _discover_case_directories(
    root: Path, replicates: Tuple[str, ...]
) -> Sequence[Path]:
    directories: List[Path] = []
    for condition_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for replicate in replicates:
            candidate = condition_directory / replicate
            if candidate.is_dir():
                directories.append(candidate)
    return directories


def load_rcaeval_cases(
    raw_path: str, profile: RCAEvalDatasetProfile
) -> RCAEvalAdapterResult:
    """Load locally available RCAEval cases without exposing label paths.

    Prediction-visible telemetry URIs contain only opaque case IDs. Actual local
    paths remain in the trusted sources sidecar used by data materialization.
    """

    root = Path(raw_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))

    inputs = []
    labels = []
    sources = []
    excluded = []
    for case_directory in _discover_case_directories(root, profile.replicates):
        relative = str(case_directory.relative_to(root))
        required = {
            "metrics": case_directory / "metrics.csv",
            "logs": case_directory / "logs.csv",
            "traces": case_directory / "traces.csv",
            "inject_time": case_directory / "inject_time.txt",
        }
        missing = sorted(name for name, path in required.items() if not path.is_file())
        if missing:
            excluded.append(
                RCAEvalExcludedCase(relative, "missing_files:{}".format(",".join(missing)))
            )
            continue

        try:
            root_service, fault_type = _parse_condition(case_directory.parent.name)
            anchor_time = int(required["inject_time"].read_text(encoding="utf-8").strip())
            services = _candidate_services(case_directory, profile.auxiliary_entities)
        except (OSError, StopIteration, TypeError, ValueError) as exc:
            excluded.append(
                RCAEvalExcludedCase(relative, "parse_error:{}".format(type(exc).__name__))
            )
            continue

        if root_service not in services:
            excluded.append(RCAEvalExcludedCase(relative, "root_not_in_candidates"))
            continue

        case_id = _opaque_case_id(profile, relative)
        base_uri = "rcaeval://{}/{}".format(profile.uri_namespace, case_id)
        inputs.append(
            RCACaseInput(
                case_id=case_id,
                dataset=profile.dataset,
                anchor_time=anchor_time,
                services=services,
                metrics=TelemetryRef(
                    uri=base_uri + "/metrics",
                    format="csv-wide",
                    timestamp_column="time",
                    metadata={"timestamp_unit": "s"},
                ),
                logs=TelemetryRef(
                    uri=base_uri + "/logs",
                    format="csv",
                    timestamp_column="timestamp",
                    service_column="container_name",
                    metadata={"timestamp_unit": "ns"},
                ),
                traces=TelemetryRef(
                    uri=base_uri + "/traces",
                    format="csv",
                    timestamp_column="startTime",
                    service_column="serviceName",
                    metadata={"timestamp_unit": "us"},
                ),
                topology=None,
                metadata={
                    "timestamp_unit": "s",
                    "candidate_rule": "CPU/memory entities excluding auxiliary containers",
                },
            )
        )
        labels.append(RCACaseLabel(case_id, root_service, fault_type))
        sources.append(
            RCAEvalCaseSource(
                case_id=case_id,
                relative_directory=relative,
                metrics_path=str(required["metrics"]),
                logs_path=str(required["logs"]),
                traces_path=str(required["traces"]),
                inject_time_path=str(required["inject_time"]),
            )
        )

    validate_case_collection(inputs, labels)
    return RCAEvalAdapterResult(
        inputs=tuple(inputs),
        labels=tuple(labels),
        sources=tuple(sources),
        excluded=tuple(excluded),
    )


def load_re2ob_cases(raw_path: str) -> RCAEvalAdapterResult:
    """Load RE2-OB cases. Signature frozen: the P1 manifest depends on it."""

    return load_rcaeval_cases(raw_path, RE2OB_PROFILE)


def load_re2tt_cases(raw_path: str) -> RCAEvalAdapterResult:
    """Load RE2-TT (TrainTicket) cases under the same unmodified rules."""

    return load_rcaeval_cases(raw_path, RE2TT_PROFILE)
