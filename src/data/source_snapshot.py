"""Content-addressed snapshots for raw dataset files consumed by experiments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SOURCE_SNAPSHOT_SCHEMA_VERSION = "p1_source_snapshot_v1"
_CHUNK_BYTES = 8 * 1024 * 1024


class SourceSnapshotError(ValueError):
    """Raised when a source snapshot is unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True)
class ConsumedSource:
    """One raw file used by one benchmark case."""

    case_id: str
    role: str
    path: str


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(source_root: Path, raw_path: str) -> tuple:
    path = Path(raw_path).resolve(strict=True)
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise SourceSnapshotError(
            "source path is outside the declared source root: {}".format(raw_path)
        ) from exc
    if not path.is_file():
        raise SourceSnapshotError("source path is not a regular file: {}".format(raw_path))
    return path, relative.as_posix()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            for record in records:
                rendered = _canonical_line(record)
                handle.write(rendered)
                digest.update(rendered)
                total_bytes += int(record["bytes"])
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "indexed_bytes": total_bytes,
        "rows": len(records),
        "sha256": digest.hexdigest(),
    }


def _content_identity(files: Mapping[str, Mapping[str, Any]]) -> str:
    identity = {
        filename: {
            "rows": details["rows"],
            "sha256": details["sha256"],
        }
        for filename, details in sorted(files.items())
    }
    return hashlib.sha256(_canonical_line(identity)).hexdigest()


def write_source_snapshot(
    output_directory: str,
    dataset: str,
    source_root: str,
    consumed_sources: Iterable[ConsumedSource],
    archives: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Mapping[str, Any]:
    """Hash raw inputs and write a deterministic, location-independent snapshot."""

    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise SourceSnapshotError("source_root must be a directory")

    resolved_consumed = []
    seen_case_roles = set()
    seen_paths = set()
    for source in consumed_sources:
        if not source.case_id or not source.role:
            raise SourceSnapshotError("consumed sources require case_id and role")
        key = (source.case_id, source.role)
        if key in seen_case_roles:
            raise SourceSnapshotError(
                "duplicate case/role source: {}/{}".format(*key)
            )
        path, relative = _resolve_source_path(root, source.path)
        if relative in seen_paths:
            raise SourceSnapshotError(
                "the same raw file is assigned more than once: {}".format(relative)
            )
        seen_case_roles.add(key)
        seen_paths.add(relative)
        resolved_consumed.append((source.case_id, source.role, relative, path))

    resolved_archives = []
    for raw_path in archives:
        path, relative = _resolve_source_path(root, raw_path)
        if relative in seen_paths:
            raise SourceSnapshotError(
                "archive duplicates a consumed source: {}".format(relative)
            )
        if relative in {item[0] for item in resolved_archives}:
            raise SourceSnapshotError("duplicate archive: {}".format(relative))
        resolved_archives.append((relative, path))

    resolved_consumed.sort(key=lambda item: (item[0], item[1], item[2]))
    resolved_archives.sort(key=lambda item: item[0])
    total = len(resolved_consumed) + len(resolved_archives)
    done = 0

    consumed_records = []
    role_counts = {}
    role_bytes = {}
    for case_id, role, relative, path in resolved_consumed:
        size = path.stat().st_size
        consumed_records.append(
            {
                "bytes": size,
                "case_id": case_id,
                "relative_path": relative,
                "role": role,
                "sha256": _sha256_file(path),
            }
        )
        role_counts[role] = role_counts.get(role, 0) + 1
        role_bytes[role] = role_bytes.get(role, 0) + size
        done += 1
        if progress is not None:
            progress(done, total, relative)

    archive_records = []
    for relative, path in resolved_archives:
        archive_records.append(
            {
                "bytes": path.stat().st_size,
                "relative_path": relative,
                "sha256": _sha256_file(path),
            }
        )
        done += 1
        if progress is not None:
            progress(done, total, relative)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "archives.jsonl": {
            "role": "source_archive_index",
            **_write_jsonl(output / "archives.jsonl", archive_records),
        },
        "consumed_files.jsonl": {
            "role": "consumed_dataset_file_index",
            **_write_jsonl(output / "consumed_files.jsonl", consumed_records),
        },
    }
    files["consumed_files.jsonl"]["counts_by_role"] = dict(sorted(role_counts.items()))
    files["consumed_files.jsonl"]["bytes_by_role"] = dict(sorted(role_bytes.items()))
    index = {
        "case_count": len({record["case_id"] for record in consumed_records}),
        "content_identity_sha256": _content_identity(files),
        "dataset": dataset,
        "files": files,
        "metadata": dict(metadata or {}),
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "source_root_name": root.name,
    }
    rendered = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return index


def verify_source_snapshot(
    output_directory: str,
    source_root: str,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Mapping[str, Any]:
    """Verify both snapshot indexes and every indexed raw file byte-for-byte."""

    output = Path(output_directory)
    root = Path(source_root).resolve(strict=True)
    try:
        index = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError("cannot read source snapshot manifest") from exc
    if index.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise SourceSnapshotError("unsupported source snapshot schema version")
    files = index.get("files")
    if not isinstance(files, dict):
        raise SourceSnapshotError("source snapshot files index is invalid")
    required = ("archives.jsonl", "consumed_files.jsonl")
    if any(filename not in files for filename in required):
        raise SourceSnapshotError("source snapshot is missing a required index")

    parsed_by_file = {}
    for filename in required:
        digest = hashlib.sha256()
        records = []
        try:
            with (output / filename).open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise SourceSnapshotError(
                            "{} contains a non-terminated row".format(filename)
                        )
                    digest.update(line)
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise SourceSnapshotError("snapshot rows must be JSON objects")
                    records.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceSnapshotError("cannot read {}".format(filename)) from exc
        expected = files[filename]
        if len(records) != expected.get("rows") or digest.hexdigest() != expected.get(
            "sha256"
        ):
            raise SourceSnapshotError("{} checksum or row mismatch".format(filename))
        if sum(int(record["bytes"]) for record in records) != expected.get(
            "indexed_bytes"
        ):
            raise SourceSnapshotError("{} byte count mismatch".format(filename))
        parsed_by_file[filename] = records

    if _content_identity(files) != index.get("content_identity_sha256"):
        raise SourceSnapshotError("content identity mismatch")

    all_records = parsed_by_file["consumed_files.jsonl"] + parsed_by_file[
        "archives.jsonl"
    ]
    for done, record in enumerate(all_records, start=1):
        relative_path = record["relative_path"]
        if Path(relative_path).is_absolute():
            raise SourceSnapshotError("snapshot source paths must be relative")
        path, relative = _resolve_source_path(root, str(root / relative_path))
        if relative != record["relative_path"]:
            raise SourceSnapshotError("source path is not canonical")
        if path.stat().st_size != record.get("bytes"):
            raise SourceSnapshotError(
                "source size mismatch: {}".format(record["relative_path"])
            )
        if _sha256_file(path) != record.get("sha256"):
            raise SourceSnapshotError(
                "source checksum mismatch: {}".format(record["relative_path"])
            )
        if progress is not None:
            progress(done, len(all_records), relative)
    return index
