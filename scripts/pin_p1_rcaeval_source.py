#!/usr/bin/env python3
"""Pin the RE2-OB archive and every raw file consumed by the P1 benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.source_snapshot import (
    ConsumedSource,
    verify_source_snapshot,
    write_source_snapshot,
)


ROLE_FIELDS = {
    "inject_time": "inject_time_path",
    "logs": "logs_path",
    "metrics": "metrics_path",
    "traces": "traces_path",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sources(path: Path) -> tuple:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON at {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("source rows must be JSON objects")
            records.append(record)
    return tuple(records)


def _consumed_sources(records: tuple) -> tuple:
    consumed = []
    case_ids = set()
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every source row requires case_id")
        if case_id in case_ids:
            raise ValueError("duplicate case_id: {}".format(case_id))
        case_ids.add(case_id)
        for role, field in ROLE_FIELDS.items():
            path = record.get(field)
            if not isinstance(path, str) or not path:
                raise ValueError("{} is missing {}".format(case_id, field))
            consumed.append(ConsumedSource(case_id=case_id, role=role, path=path))
    return tuple(consumed)


def _progress(prefix: str):
    def report(done: int, total: int, relative_path: str) -> None:
        if done == 1 or done == total or done % 20 == 0:
            print(
                "{} {}/{} {}".format(prefix, done, total, relative_path),
                flush=True,
            )

    return report


def _summary(index: Mapping[str, object], manifest_path: Path) -> Mapping[str, object]:
    files = index["files"]
    consumed = files["consumed_files.jsonl"]
    archives = files["archives.jsonl"]
    return {
        "archive_bytes": archives["indexed_bytes"],
        "archive_count": archives["rows"],
        "case_count": index["case_count"],
        "consumed_bytes": consumed["indexed_bytes"],
        "consumed_file_count": consumed["rows"],
        "content_identity_sha256": index["content_identity_sha256"],
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.verify_only:
        index = verify_source_snapshot(
            str(output), args.source_root, progress=_progress("verify")
        )
    else:
        case_manifest = Path(args.case_manifest)
        records = _read_sources(case_manifest / "sources.jsonl")
        consumed = _consumed_sources(records)
        manifest_sha = _sha256(case_manifest / "manifest.json")
        sources_sha = _sha256(case_manifest / "sources.jsonl")
        index = write_source_snapshot(
            output_directory=str(output),
            dataset="RCAEval-RE2-OB",
            source_root=args.source_root,
            consumed_sources=consumed,
            archives=args.archive,
            metadata={
                "case_manifest_sha256": manifest_sha,
                "case_sources_sha256": sources_sha,
                "scope": (
                    "one source archive plus the exact metrics/logs/traces/"
                    "inject-time files referenced by the 90-case P1 manifest"
                ),
            },
            progress=_progress("hash"),
        )
    print(
        json.dumps(
            _summary(index, output / "manifest.json"),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
