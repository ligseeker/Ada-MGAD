"""Label-free, service-level feature interfaces for P2 RCA experiments."""

from .event_checkpoint import (
    EVENT_CHECKPOINT_SCHEMA_VERSION,
    EventCheckpointError,
    load_event_checkpoint,
    write_event_checkpoint,
)

from .event_summary import (
    DEFAULT_EVENT_ONSET_SECONDS,
    PreparedLogStream,
    PreparedTraceStream,
    RE2_TRACE_SERVICE_ALIASES,
    log_feature_names,
    log_stream_features,
    normalize_trace_service,
    trace_feature_names,
    trace_stream_features,
)
from .metric_summary import (
    DEFAULT_ONSET_SECONDS,
    MetricContrast,
    MetricContrastBatch,
    MetricFeatureAccumulator,
    aggregate_metric_contrasts,
    metric_feature_names,
    metric_series_contrasts,
    metric_series_contrasts_many,
)
from .schema import (
    FEATURE_BUNDLE_SCHEMA_VERSION,
    FeatureRow,
    FeatureSchemaError,
    load_feature_matrices,
    read_feature_bundle,
    verify_feature_bundle,
    write_feature_bundle,
)

__all__ = [
    "DEFAULT_ONSET_SECONDS",
    "DEFAULT_EVENT_ONSET_SECONDS",
    "EVENT_CHECKPOINT_SCHEMA_VERSION",
    "EventCheckpointError",
    "FEATURE_BUNDLE_SCHEMA_VERSION",
    "FeatureRow",
    "FeatureSchemaError",
    "MetricContrast",
    "MetricContrastBatch",
    "MetricFeatureAccumulator",
    "PreparedLogStream",
    "PreparedTraceStream",
    "RE2_TRACE_SERVICE_ALIASES",
    "aggregate_metric_contrasts",
    "metric_feature_names",
    "metric_series_contrasts",
    "metric_series_contrasts_many",
    "load_feature_matrices",
    "load_event_checkpoint",
    "log_feature_names",
    "log_stream_features",
    "normalize_trace_service",
    "read_feature_bundle",
    "verify_feature_bundle",
    "write_feature_bundle",
    "write_event_checkpoint",
    "trace_feature_names",
    "trace_stream_features",
]
