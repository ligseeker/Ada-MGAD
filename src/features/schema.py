"""Deterministic Label Firewall boundary for P2 service-level features."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np

from src.data.schema import RCACaseInput, assert_label_free


FEATURE_BUNDLE_SCHEMA_VERSION = "p2_feature_bundle_v2"
_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {"faulttype", "groundtruth", "label", "rootcause", "rootservice", "target"}
)


class FeatureSchemaError(ValueError):
    """Raised when P2 features are incomplete, non-finite, or label-bearing."""


@lru_cache(maxsize=128)
def _validate_feature_names(feature_names: Tuple[str, ...]) -> None:
    if not feature_names:
        raise FeatureSchemaError("feature_names must be a non-empty tuple")
    if len(set(feature_names)) != len(feature_names):
        raise FeatureSchemaError("feature_names must be unique")
    for name in feature_names:
        _require_text(name, "feature name")
        normalized = _normalized(name)
        if any(token in normalized for token in _FORBIDDEN_FEATURE_TOKENS):
            raise FeatureSchemaError(
                "Label Firewall rejected feature name {!r}".format(name)
            )


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FeatureSchemaError("{} must be non-empty trimmed text".format(field_name))
    return value


@dataclass(frozen=True)
class FeatureRow:
    """One fixed-width, label-free feature vector for a candidate service."""

    case_id: str
    service: str
    extractor: str
    feature_names: Tuple[str, ...]
    values: Tuple[float, ...]
    observed: Tuple[bool, ...]

    def __post_init__(self) -> None:
        _require_text(self.case_id, "FeatureRow.case_id")
        _require_text(self.service, "FeatureRow.service")
        _require_text(self.extractor, "FeatureRow.extractor")
        if not isinstance(self.feature_names, tuple):
            raise FeatureSchemaError("feature_names must be a tuple")
        _validate_feature_names(self.feature_names)
        if not isinstance(self.values, tuple) or not isinstance(self.observed, tuple):
            raise FeatureSchemaError("values and observed must be tuples")
        if not (
            len(self.feature_names) == len(self.values) == len(self.observed)
        ):
            raise FeatureSchemaError("feature names, values, and masks must align")
        for index, (value, observed) in enumerate(zip(self.values, self.observed)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FeatureSchemaError("feature values must be numeric")
            if not math.isfinite(float(value)):
                raise FeatureSchemaError("feature values must be finite")
            if not isinstance(observed, bool):
                raise FeatureSchemaError("observed masks must be bool")
            if not observed and float(value) != 0.0:
                raise FeatureSchemaError(
                    "masked feature values must be zero at index {}".format(index)
                )


def _expected_pairs(inputs: Sequence[RCACaseInput]) -> Tuple[Tuple[str, str], ...]:
    if not inputs:
        raise FeatureSchemaError("at least one input case is required")
    seen_ids = set()
    pairs = []
    for case_input in inputs:
        assert_label_free(case_input)
        if case_input.case_id in seen_ids:
            raise FeatureSchemaError("input case_id must be unique")
        seen_ids.add(case_input.case_id)
        pairs.extend((case_input.case_id, service) for service in case_input.services)
    return tuple(sorted(pairs))


def _canonical_line(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        record, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_feature_bundle(
    output_directory: str,
    dataset: str,
    inputs: Sequence[RCACaseInput],
    rows: Iterable[FeatureRow],
    extractor_config: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write deterministic binary matrices plus a label-free row index."""

    _require_text(dataset, "dataset")
    pairs = _expected_pairs(inputs)
    position_by_pair = {pair: index for index, pair in enumerate(pairs)}
    feature_names = None
    extractor = None
    values = None
    observed = None
    seen_pairs = set()
    for row in rows:
        pair = (row.case_id, row.service)
        if pair not in position_by_pair:
            raise FeatureSchemaError("unexpected feature row: {}".format(pair))
        if pair in seen_pairs:
            raise FeatureSchemaError("duplicate feature row: {}".format(pair))
        seen_pairs.add(pair)
        if feature_names is None:
            feature_names = row.feature_names
            extractor = row.extractor
            values = np.empty((len(pairs), len(feature_names)), dtype="<f4")
            observed = np.empty((len(pairs), len(feature_names)), dtype=np.bool_)
        if row.feature_names != feature_names:
            raise FeatureSchemaError("all rows must use the same feature schema")
        if row.extractor != extractor:
            raise FeatureSchemaError("all rows must use one extractor")
        index = position_by_pair[pair]
        values[index, :] = np.asarray(row.values, dtype="<f4")
        observed[index, :] = np.asarray(row.observed, dtype=np.bool_)
    if seen_pairs != set(pairs):
        missing = sorted(set(pairs) - seen_pairs)
        raise FeatureSchemaError(
            "feature coverage mismatch; missing={}".format(missing[:5])
        )
    if feature_names is None or values is None or observed is None:
        raise FeatureSchemaError("feature rows must not be empty")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for case_id, service in pairs:
                rendered = _canonical_line(
                    {"case_id": case_id, "extractor": extractor, "service": service}
                )
                handle.write(rendered)
                digest.update(rendered)
        temporary.replace(index_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    values_path = output / "values.npy"
    observed_path = output / "observed.npy"
    _write_npy(values_path, values)
    _write_npy(observed_path, observed)

    manifest = {
        "case_count": len(inputs),
        "dataset": dataset,
        "extractor": extractor,
        "extractor_config": dict(extractor_config),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "files": {
            "index.jsonl": {
                "role": "prediction_feature_index",
                "rows": len(pairs),
                "sha256": digest.hexdigest(),
            },
            "observed.npy": {
                "bytes": observed_path.stat().st_size,
                "dtype": "bool",
                "role": "prediction_feature_mask",
                "sha256": _sha256(observed_path),
                "shape": list(observed.shape),
            },
            "values.npy": {
                "bytes": values_path.stat().st_size,
                "dtype": "float32",
                "role": "prediction_feature_values",
                "sha256": _sha256(values_path),
                "shape": list(values.shape),
            },
        },
        "schema_version": FEATURE_BUNDLE_SCHEMA_VERSION,
        "service_row_count": len(pairs),
        "source": dict(source_bindings),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _read_index(path: Path) -> tuple:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    raise FeatureSchemaError("feature index row is not newline terminated")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise FeatureSchemaError("feature index rows must be objects")
                if _FORBIDDEN_FEATURE_TOKENS.intersection(
                    _normalized(key) for key in record
                ):
                    raise FeatureSchemaError("feature index contains a sensitive key")
                if set(record) != {"case_id", "extractor", "service"}:
                    raise FeatureSchemaError("feature index has unexpected fields")
                _require_text(record["case_id"], "case_id")
                _require_text(record["service"], "service")
                _require_text(record["extractor"], "extractor")
                rows.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureSchemaError("cannot read feature index") from exc
    return tuple(rows)


def _load_matrix(path: Path, expected: Mapping[str, Any], dtype) -> np.ndarray:
    if not path.is_file() or _sha256(path) != expected.get("sha256"):
        raise FeatureSchemaError("feature matrix checksum mismatch: {}".format(path))
    if path.stat().st_size != expected.get("bytes"):
        raise FeatureSchemaError("feature matrix byte mismatch: {}".format(path))
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FeatureSchemaError("cannot load feature matrix: {}".format(path)) from exc
    if array.dtype != np.dtype(dtype) or list(array.shape) != expected.get("shape"):
        raise FeatureSchemaError("feature matrix dtype or shape mismatch: {}".format(path))
    return array


def verify_feature_bundle(
    output_directory: str,
    inputs: Sequence[RCACaseInput] = (),
) -> Mapping[str, Any]:
    """Verify index/matrix bytes, finite values, masks, and optional coverage."""

    output = Path(output_directory)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureSchemaError("cannot read feature manifest") from exc
    if manifest.get("schema_version") != FEATURE_BUNDLE_SCHEMA_VERSION:
        raise FeatureSchemaError("unsupported feature bundle schema")
    files = manifest.get("files", {})
    if set(files) != {"index.jsonl", "observed.npy", "values.npy"}:
        raise FeatureSchemaError("feature manifest file set mismatch")
    index_path = output / "index.jsonl"
    if not index_path.is_file() or _sha256(index_path) != files["index.jsonl"].get(
        "sha256"
    ):
        raise FeatureSchemaError("feature index checksum mismatch")
    index = _read_index(index_path)
    if len(index) != files["index.jsonl"].get("rows") or len(index) != manifest.get(
        "service_row_count"
    ):
        raise FeatureSchemaError("feature index row count mismatch")
    pairs = [(row["case_id"], row["service"]) for row in index]
    if len(set(pairs)) != len(pairs) or pairs != sorted(pairs):
        raise FeatureSchemaError("feature index pairs must be unique and sorted")
    if any(row["extractor"] != manifest.get("extractor") for row in index):
        raise FeatureSchemaError("feature index extractor mismatch")

    values = _load_matrix(output / "values.npy", files["values.npy"], "<f4")
    observed = _load_matrix(
        output / "observed.npy", files["observed.npy"], np.bool_
    )
    if values.shape != observed.shape or values.shape != (
        len(index),
        manifest.get("feature_count"),
    ):
        raise FeatureSchemaError("feature matrix dimensions do not match manifest")
    names = tuple(manifest.get("feature_names", ()))
    FeatureRow(
        "schema-check",
        "schema-check",
        manifest.get("extractor"),
        names,
        (0.0,) * len(names),
        (False,) * len(names),
    )
    for start in range(0, len(index), 10000):
        block_values = np.asarray(values[start : start + 10000])
        block_observed = np.asarray(observed[start : start + 10000])
        if not np.isfinite(block_values).all():
            raise FeatureSchemaError("feature values contain non-finite values")
        if np.any(block_values[~block_observed] != 0.0):
            raise FeatureSchemaError("masked feature values must be zero")
    if inputs:
        expected_pairs = _expected_pairs(inputs)
        if tuple(pairs) != expected_pairs:
            raise FeatureSchemaError("feature coverage does not match inputs")
        if len(inputs) != manifest.get("case_count"):
            raise FeatureSchemaError("feature case count mismatch")
    return manifest


def read_feature_bundle(output_directory: str) -> tuple:
    """Read a verified feature bundle into row objects for model development."""

    output = Path(output_directory)
    manifest = verify_feature_bundle(output_directory)
    index = _read_index(output / "index.jsonl")
    values = np.load(output / "values.npy", allow_pickle=False)
    observed = np.load(output / "observed.npy", allow_pickle=False)
    names = tuple(manifest["feature_names"])
    rows = tuple(
        FeatureRow(
            case_id=record["case_id"],
            service=record["service"],
            extractor=record["extractor"],
            feature_names=names,
            values=tuple(float(value) for value in values[index_value]),
            observed=tuple(bool(value) for value in observed[index_value]),
        )
        for index_value, record in enumerate(index)
    )
    return manifest, rows


def load_feature_matrices(output_directory: str) -> tuple:
    """Return a verified row index and memory-mapped matrices for model fitting."""

    output = Path(output_directory)
    manifest = verify_feature_bundle(output_directory)
    index = _read_index(output / "index.jsonl")
    values = np.load(output / "values.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(
        output / "observed.npy", mmap_mode="r", allow_pickle=False
    )
    return manifest, index, values, observed
