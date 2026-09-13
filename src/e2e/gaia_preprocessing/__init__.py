"""GAIA preprocessing V2 public interface.

The V2 path is deliberately schema-first: transforms may consume a frozen
schema, but they must never select features or fit statistics from Test data.
"""

from .schema import (
    FrozenPreprocessingSchema,
    load_frozen_preprocessing_schema,
    verify_frozen_schema_sources,
)
from .materialize import validate_transformed_modalities

__all__ = [
    "FrozenPreprocessingSchema",
    "load_frozen_preprocessing_schema",
    "validate_transformed_modalities",
    "verify_frozen_schema_sources",
]
