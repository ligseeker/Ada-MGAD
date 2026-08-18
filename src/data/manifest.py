"""Deterministic, label-separated manifests for the P1 RCA benchmark."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.data.schema import (
    RCACaseInput,
    RCACaseLabel,
    TelemetryRef,
    TopologyRef,
    validate_case_collection,
)


MANIFEST_SCHEMA_VERSION = "p1_rca_manifest_v1"


class ManifestIntegrityError(ValueError):
    """Raised when a manifest bundle is incomplete or has changed bytes."""


def _telemetry_record(reference: Optional[TelemetryRef]) -> Optional[Mapping[str, Any]]:
    if reference is None:
        return None
    return {
        "format": reference.format,
        "metadata": dict(reference.metadata),
        "service_column": reference.service_column,
        "timestamp_column": reference.timestamp_column,
        "uri": reference.uri,
    }


def _topology_record(reference: Optional[TopologyRef]) -> Optional[Mapping[str, Any]]:
    if reference is None:
        return None
    return {
        "directed": reference.directed,
        "format": reference.format,
        "metadata": dict(reference.metadata),
        "uri": reference.uri,
    }


def input_record(case: RCACaseInput) -> Mapping[str, Any]:
    """Serialize only fields available to a predictor."""

    return {
        "anchor_time": case.anchor_time,
        "case_id": case.case_id,
        "dataset": case.dataset,
        "logs": _telemetry_record(case.logs),
        "metadata": dict(case.metadata),
        "metrics": _telemetry_record(case.metrics),
        "services": list(case.services),
        "topology": _topology_record(case.topology),
        "traces": _telemetry_record(case.traces),
    }


def label_record(label: RCACaseLabel) -> Mapping[str, Any]:
    """Serialize evaluation/training-only fields."""

    return {
        "case_id": label.case_id,
        "fault_type": label.fault_type,
        "root_service": label.root_service,
    }


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


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    rows = 0
    try:
        with temporary.open("wb") as handle:
            for record in records:
                rendered = _canonical_line(record)
                handle.write(rendered)
                digest.update(rendered)
                rows += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"rows": rows, "sha256": digest.hexdigest()}


def write_manifest_bundle(
    output_directory: str,
    dataset: str,
    inputs: Sequence[RCACaseInput],
    labels: Sequence[RCACaseLabel],
    trusted_sidecars: Optional[
        Mapping[str, Sequence[Mapping[str, Any]]]
    ] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Write deterministic files and a checksum-bearing bundle index.

    The two mandatory files are physically separate. Additional sidecars are
    trusted artifacts and their names must not shadow either mandatory file.
    """

    validate_case_collection(inputs, labels)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    files["inputs.jsonl"] = {
        "role": "prediction_input",
        **_write_jsonl(output / "inputs.jsonl", map(input_record, inputs)),
    }
    files["labels.jsonl"] = {
        "role": "training_evaluation_label",
        **_write_jsonl(output / "labels.jsonl", map(label_record, labels)),
    }

    for filename, records in sorted((trusted_sidecars or {}).items()):
        if (
            filename in files
            or Path(filename).name != filename
            or not filename.endswith(".jsonl")
        ):
            raise ManifestIntegrityError(
                "sidecar names must be simple, unique .jsonl filenames"
            )
        files[filename] = {
            "role": "trusted_sidecar",
            **_write_jsonl(output / filename, records),
        }

    index = {
        "case_count": len(inputs),
        "dataset": dataset,
        "files": files,
        "metadata": dict(metadata or {}),
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    rendered = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    index_path = output / "manifest.json"
    temporary = index_path.with_name(index_path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(index_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return index


def verify_manifest_bundle(output_directory: str) -> Mapping[str, Any]:
    """Recompute every indexed row count and SHA-256 checksum."""

    output = Path(output_directory)
    index_path = output / "manifest.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestIntegrityError("cannot read manifest.json") from exc
    if index.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestIntegrityError("unsupported manifest schema version")
    files = index.get("files")
    if not isinstance(files, dict):
        raise ManifestIntegrityError("manifest files index is invalid")
    for required in ("inputs.jsonl", "labels.jsonl"):
        if required not in files:
            raise ManifestIntegrityError("manifest is missing {}".format(required))

    for filename, expected in files.items():
        path = output / filename
        digest = hashlib.sha256()
        rows = 0
        try:
            with path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise ManifestIntegrityError(
                            "{} contains a non-terminated JSONL row".format(filename)
                        )
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ManifestIntegrityError(
                            "{} contains invalid JSON".format(filename)
                        ) from exc
                    if not isinstance(parsed, dict):
                        raise ManifestIntegrityError(
                            "{} rows must be JSON objects".format(filename)
                        )
                    digest.update(line)
                    rows += 1
        except OSError as exc:
            raise ManifestIntegrityError("cannot read {}".format(filename)) from exc
        if rows != expected.get("rows") or digest.hexdigest() != expected.get("sha256"):
            raise ManifestIntegrityError(
                "{} row count or checksum mismatch".format(filename)
            )
    if files["inputs.jsonl"]["rows"] != index.get("case_count"):
        raise ManifestIntegrityError("case_count does not match inputs.jsonl")
    if files["labels.jsonl"]["rows"] != index.get("case_count"):
        raise ManifestIntegrityError("case_count does not match labels.jsonl")
    return index
