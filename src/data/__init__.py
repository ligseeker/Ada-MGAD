"""Dataset-independent schemas for standalone RCA."""

from .schema import (
    RCACaseInput,
    RCACaseLabel,
    SchemaValidationError,
    TelemetryRef,
    TopologyRef,
    assert_label_free,
    validate_case_collection,
)

from .gaia import load_gaia_cases
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestIntegrityError,
    verify_manifest_bundle,
    write_manifest_bundle,
)
from .rcaeval import load_re2ob_cases
from .split import (
    CaseGroup,
    SplitAssignment,
    SplitIntegrityError,
    build_overlap_groups,
    build_singleton_groups,
    validate_split_integrity,
)

__all__ = [
    "RCACaseInput",
    "RCACaseLabel",
    "SchemaValidationError",
    "TelemetryRef",
    "TopologyRef",
    "assert_label_free",
    "validate_case_collection",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestIntegrityError",
    "verify_manifest_bundle",
    "write_manifest_bundle",
    "CaseGroup",
    "SplitAssignment",
    "SplitIntegrityError",
    "build_overlap_groups",
    "build_singleton_groups",
    "validate_split_integrity",
    "load_gaia_cases",
    "load_re2ob_cases",
]
