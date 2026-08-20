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
from .event_purity import (
    EventPurityFlag,
    LabeledEventInterval,
    build_event_purity_flags,
)
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestIntegrityError,
    read_manifest_cases,
    verify_manifest_bundle,
    write_manifest_bundle,
)
from .rcaeval import (
    DATASET_PROFILES,
    RE2OB_PROFILE,
    RE2TT_PROFILE,
    RCAEvalDatasetProfile,
    load_rcaeval_cases,
    load_re2ob_cases,
    load_re2tt_cases,
)
from .source_snapshot import (
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
    ConsumedSource,
    SourceSnapshotError,
    verify_source_snapshot,
    write_source_snapshot,
)
from .split import (
    CaseGroup,
    CaseInterval,
    SplitAssignment,
    SplitIntegrityError,
    assign_balanced_group_folds,
    assign_contiguous_group_folds,
    build_interval_overlap_groups,
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
    "read_manifest_cases",
    "verify_manifest_bundle",
    "write_manifest_bundle",
    "EventPurityFlag",
    "LabeledEventInterval",
    "build_event_purity_flags",
    "CaseGroup",
    "CaseInterval",
    "SplitAssignment",
    "SplitIntegrityError",
    "assign_balanced_group_folds",
    "assign_contiguous_group_folds",
    "build_interval_overlap_groups",
    "build_overlap_groups",
    "build_singleton_groups",
    "validate_split_integrity",
    "load_gaia_cases",
    "load_re2ob_cases",
    "load_re2tt_cases",
    "load_rcaeval_cases",
    "DATASET_PROFILES",
    "RE2OB_PROFILE",
    "RE2TT_PROFILE",
    "RCAEvalDatasetProfile",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "ConsumedSource",
    "SourceSnapshotError",
    "verify_source_snapshot",
    "write_source_snapshot",
]
