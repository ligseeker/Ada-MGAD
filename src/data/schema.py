"""Logical schema and Label Firewall for event-level, service-level RCA.

The prediction-facing object is deliberately separate from :class:`RCACaseLabel`.
Code that extracts features or produces rankings should accept only
``RCACaseInput`` instances. Labels are reserved for split construction,
training loss, evaluation, and error analysis.
"""

from dataclasses import dataclass, field
import math
from numbers import Real
import re
from typing import Any, Mapping, Optional, Sequence, Tuple


class SchemaValidationError(ValueError):
    """Raised when a case violates the unified RCA schema."""


# Normalization below also catches spelling variants such as ``root-service``
# and ``faultType``. The list is intentionally strict because RCACaseInput
# metadata is prediction-visible.
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "faultlabel",
        "faulttype",
        "groundtruth",
        "label",
        "labels",
        "rootcause",
        "rootindicator",
        "rootlabel",
        "rootservice",
        "target",
        "testrootfrequency",
    }
)


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _validate_label_free_value(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_METADATA_KEYS:
                raise SchemaValidationError(
                    "Label Firewall rejected prediction-visible metadata key "
                    "{!r} at {}".format(key, path)
                )
            _validate_label_free_value(nested, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_label_free_value(nested, "{}[{}]".format(path, index))


def _validated_metadata(metadata: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise SchemaValidationError("{} must be a mapping".format(path))
    copied = dict(metadata)
    _validate_label_free_value(copied, path)
    return copied


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError("{} must be a non-empty string".format(field_name))
    if value != value.strip():
        raise SchemaValidationError("{} must not contain surrounding whitespace".format(field_name))
    return value


@dataclass(frozen=True)
class TelemetryRef:
    """Traceable reference to one case's modality data.

    ``uri`` may be a local path or another stable artifact identifier. It is a
    reference only: constructing the schema never reads the underlying data.
    """

    uri: str
    format: Optional[str] = None
    timestamp_column: Optional[str] = None
    service_column: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_text(self.uri, "TelemetryRef.uri")
        if self.format is not None:
            _require_nonempty_text(self.format, "TelemetryRef.format")
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, "TelemetryRef.metadata"),
        )


@dataclass(frozen=True)
class TopologyRef:
    """Traceable reference to topology available at prediction time."""

    uri: str
    format: Optional[str] = None
    directed: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_text(self.uri, "TopologyRef.uri")
        if self.format is not None:
            _require_nonempty_text(self.format, "TopologyRef.format")
        if self.directed is not None and not isinstance(self.directed, bool):
            raise SchemaValidationError("TopologyRef.directed must be bool or None")
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, "TopologyRef.metadata"),
        )


@dataclass(frozen=True)
class RCACaseInput:
    """All information that an RCA predictor may access for one fault case."""

    case_id: str
    dataset: str
    anchor_time: Real
    services: Tuple[str, ...]
    metrics: Optional[TelemetryRef] = None
    logs: Optional[TelemetryRef] = None
    traces: Optional[TelemetryRef] = None
    topology: Optional[TopologyRef] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_text(self.case_id, "RCACaseInput.case_id")
        _require_nonempty_text(self.dataset, "RCACaseInput.dataset")

        if isinstance(self.anchor_time, bool) or not isinstance(self.anchor_time, Real):
            raise SchemaValidationError("anchor_time must be a finite int or float")
        if not math.isfinite(float(self.anchor_time)):
            raise SchemaValidationError("anchor_time must be finite")

        if not isinstance(self.services, tuple):
            raise SchemaValidationError(
                "services must be a tuple with adapter-defined deterministic order"
            )
        if not self.services:
            raise SchemaValidationError("services must not be empty")
        for index, service in enumerate(self.services):
            _require_nonempty_text(service, "services[{}]".format(index))
        if len(set(self.services)) != len(self.services):
            raise SchemaValidationError("services must not contain duplicates")

        expected_types = (
            ("metrics", self.metrics, TelemetryRef),
            ("logs", self.logs, TelemetryRef),
            ("traces", self.traces, TelemetryRef),
            ("topology", self.topology, TopologyRef),
        )
        for name, value, expected_type in expected_types:
            if value is not None and not isinstance(value, expected_type):
                raise SchemaValidationError(
                    "{} must be {} or None".format(name, expected_type.__name__)
                )

        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, "RCACaseInput.metadata"),
        )


@dataclass(frozen=True)
class RCACaseLabel:
    """Evaluation/training-only fields for one RCA case."""

    case_id: str
    root_service: str
    fault_type: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.case_id, "RCACaseLabel.case_id")
        _require_nonempty_text(self.root_service, "RCACaseLabel.root_service")
        if self.fault_type is not None:
            _require_nonempty_text(self.fault_type, "RCACaseLabel.fault_type")


def assert_label_free(case_input: RCACaseInput) -> None:
    """Re-run Label Firewall validation at a feature/prediction boundary."""

    if not isinstance(case_input, RCACaseInput):
        raise TypeError("prediction code must receive RCACaseInput, not a combined case")
    _validate_label_free_value(case_input.metadata, "RCACaseInput.metadata")
    for name in ("metrics", "logs", "traces", "topology"):
        reference = getattr(case_input, name)
        if reference is not None:
            _validate_label_free_value(
                reference.metadata,
                "RCACaseInput.{}.metadata".format(name),
            )


def validate_case_collection(
    inputs: Sequence[RCACaseInput],
    labels: Optional[Sequence[RCACaseLabel]] = None,
) -> None:
    """Validate collection-level identity and root-candidate invariants."""

    input_ids = [case.case_id for case in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise SchemaValidationError("case_id must be unique across all inputs")
    for case in inputs:
        assert_label_free(case)

    if labels is None:
        return

    label_ids = [label.case_id for label in labels]
    if len(set(label_ids)) != len(label_ids):
        raise SchemaValidationError("case_id must be unique across all labels")
    if set(input_ids) != set(label_ids):
        missing = sorted(set(input_ids) - set(label_ids))
        extra = sorted(set(label_ids) - set(input_ids))
        raise SchemaValidationError(
            "input/label case_id mismatch; missing labels={}, extra labels={}".format(
                missing, extra
            )
        )

    inputs_by_id = {case.case_id: case for case in inputs}
    for label in labels:
        if label.root_service not in inputs_by_id[label.case_id].services:
            raise SchemaValidationError(
                "root_service {!r} is not in services for case {!r}".format(
                    label.root_service, label.case_id
                )
            )
