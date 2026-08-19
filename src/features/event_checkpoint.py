"""Recoverable, checksum-bound checkpoints for full event feature extraction."""

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np


EVENT_CHECKPOINT_SCHEMA_VERSION = "p2_event_checkpoint_v1"


class EventCheckpointError(ValueError):
    """Raised when an event extraction checkpoint is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_line(record) -> bytes:
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


def _write_json(path: Path, record) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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


def _normalized_pairs(pairs: Sequence[Tuple[str, str]]) -> tuple:
    result = tuple((str(case_id), str(service)) for case_id, service in pairs)
    if not result or any(not case_id or not service for case_id, service in result):
        raise EventCheckpointError("checkpoint row pairs must be non-empty text")
    if len(set(result)) != len(result):
        raise EventCheckpointError("checkpoint row pairs must be unique")
    return result


def write_event_checkpoint(
    output_directory: str,
    checkpoint_id: str,
    extractor: str,
    feature_names: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
    values,
    observed,
    source_binding: Mapping[str, object],
    extraction_stats: Mapping[str, object],
) -> Mapping[str, object]:
    """Write matrices atomically and publish the manifest last."""

    if not checkpoint_id or not extractor:
        raise EventCheckpointError("checkpoint id and extractor are required")
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise EventCheckpointError("checkpoint feature names must be unique")
    row_pairs = _normalized_pairs(pairs)
    value_array = np.asarray(values, dtype="<f4")
    observed_array = np.asarray(observed, dtype=np.bool_)
    expected_shape = (len(row_pairs), len(names))
    if value_array.shape != expected_shape or observed_array.shape != expected_shape:
        raise EventCheckpointError("checkpoint matrix shape mismatch")
    if not np.isfinite(value_array).all():
        raise EventCheckpointError("checkpoint values must be finite")
    if np.any(value_array[~observed_array] != 0.0):
        raise EventCheckpointError("masked checkpoint values must be zero")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    index_digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for case_id, service in row_pairs:
                rendered = _canonical_line(
                    {"case_id": case_id, "service": service}
                )
                handle.write(rendered)
                index_digest.update(rendered)
        temporary.replace(index_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    values_path = output / "values.npy"
    observed_path = output / "observed.npy"
    _write_npy(values_path, value_array)
    _write_npy(observed_path, observed_array)
    manifest = {
        "checkpoint_id": checkpoint_id,
        "extractor": extractor,
        "extraction_stats": dict(extraction_stats),
        "feature_names": list(names),
        "files": {
            "index.jsonl": {
                "bytes": index_path.stat().st_size,
                "sha256": index_digest.hexdigest(),
            },
            "observed.npy": {
                "bytes": observed_path.stat().st_size,
                "sha256": _sha256(observed_path),
                "shape": list(observed_array.shape),
            },
            "values.npy": {
                "bytes": values_path.stat().st_size,
                "sha256": _sha256(values_path),
                "shape": list(value_array.shape),
            },
        },
        "row_count": len(row_pairs),
        "schema_version": EVENT_CHECKPOINT_SCHEMA_VERSION,
        "source_binding": dict(source_binding),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _read_pairs(path: Path) -> tuple:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise EventCheckpointError("checkpoint index is not newline terminated")
            record = json.loads(line)
            if set(record) != {"case_id", "service"}:
                raise EventCheckpointError("checkpoint index fields changed")
            result.append((record["case_id"], record["service"]))
    return _normalized_pairs(result)


def load_event_checkpoint(
    output_directory: str,
    checkpoint_id: str,
    extractor: str,
    feature_names: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
    source_binding: Mapping[str, object],
) -> tuple:
    """Verify and load a checkpoint, failing closed on any source drift."""

    output = Path(output_directory)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventCheckpointError("cannot read checkpoint manifest") from exc
    if manifest.get("schema_version") != EVENT_CHECKPOINT_SCHEMA_VERSION:
        raise EventCheckpointError("unsupported checkpoint schema")
    expected = {
        "checkpoint_id": checkpoint_id,
        "extractor": extractor,
        "feature_names": list(feature_names),
        "row_count": len(pairs),
        "source_binding": dict(source_binding),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise EventCheckpointError("checkpoint {} mismatch".format(key))
    files = manifest.get("files", {})
    if set(files) != {"index.jsonl", "observed.npy", "values.npy"}:
        raise EventCheckpointError("checkpoint file set mismatch")
    for name, metadata in files.items():
        path = output / name
        if (
            not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or _sha256(path) != metadata.get("sha256")
        ):
            raise EventCheckpointError("checkpoint file checksum mismatch: {}".format(name))
    expected_pairs = _normalized_pairs(pairs)
    if _read_pairs(output / "index.jsonl") != expected_pairs:
        raise EventCheckpointError("checkpoint row identity mismatch")
    try:
        values = np.load(output / "values.npy", mmap_mode="r", allow_pickle=False)
        observed = np.load(
            output / "observed.npy", mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise EventCheckpointError("cannot load checkpoint matrices") from exc
    shape = (len(expected_pairs), len(feature_names))
    if (
        values.dtype != np.dtype("<f4")
        or observed.dtype != np.dtype(np.bool_)
        or values.shape != shape
        or observed.shape != shape
        or files["values.npy"].get("shape") != list(shape)
        or files["observed.npy"].get("shape") != list(shape)
    ):
        raise EventCheckpointError("checkpoint matrix metadata mismatch")
    if not np.isfinite(values).all() or np.any(values[~observed] != 0.0):
        raise EventCheckpointError("checkpoint matrix content is invalid")
    return values, observed, manifest
