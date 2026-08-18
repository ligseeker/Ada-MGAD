#!/usr/bin/env python
"""Diagnose GAIA split groups after expanding event-level telemetry contexts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    CaseInterval,
    build_interval_overlap_groups,
    verify_manifest_bundle,
)


SCHEMA_VERSION = "p1_context_group_diagnostics_v1"


def _parse_windows(value: str) -> Sequence[int]:
    windows = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not windows or any(window <= 0 for window in windows):
        raise argparse.ArgumentTypeError("windows must be positive integers")
    return windows


def _read_jsonl(path: Path) -> Sequence[Mapping[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{} contains invalid JSON at line {}".format(path, line_number)
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("{} line {} is not an object".format(path, line_number))
            records.append(record)
    return tuple(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_summary(groups: Iterable[object]) -> Mapping[str, object]:
    rows = tuple(groups)
    sizes_by_group: Dict[str, int] = {}
    for row in rows:
        sizes_by_group[str(row.group_id)] = int(row.group_size)
    sizes = tuple(sizes_by_group.values())
    non_singletons = tuple(size for size in sizes if size > 1)
    return {
        "cases": len(rows),
        "groups": len(sizes),
        "non_singleton_groups": len(non_singletons),
        "cases_in_non_singleton_groups": sum(non_singletons),
        "largest_group": max(sizes, default=0),
        "largest_group_ratio": max(sizes, default=0) / len(rows) if rows else None,
        "direct_overlap_pairs": sum(int(row.overlap_degree) for row in rows) // 2,
        "maximum_direct_overlap_degree": max(
            (int(row.overlap_degree) for row in rows), default=0
        ),
    }


def diagnose(
    manifest_directory: Path,
    windows_seconds: Sequence[int],
) -> Mapping[str, object]:
    index = verify_manifest_bundle(str(manifest_directory))
    inputs = {
        str(row["case_id"]): row
        for row in _read_jsonl(manifest_directory / "inputs.jsonl")
    }
    audit = {
        str(row["case_id"]): row
        for row in _read_jsonl(manifest_directory / "event_audit.jsonl")
    }
    if len(inputs) != index["case_count"] or set(inputs) != set(audit):
        raise ValueError("inputs/event_audit case coverage mismatch")

    for case_id, row in audit.items():
        if int(inputs[case_id]["anchor_time"]) != int(row["start_ms"]):
            raise ValueError("anchor/start mismatch for {}".format(case_id))

    by_window = {}
    for window_seconds in windows_seconds:
        radius_ms = int(window_seconds) * 1000
        context_only = []
        injection_plus_context = []
        for case_id, input_row in inputs.items():
            anchor_ms = int(input_row["anchor_time"])
            context_start = anchor_ms - radius_ms
            context_end = anchor_ms + radius_ms
            audit_row = audit[case_id]
            context_only.append(
                CaseInterval(case_id, context_start, context_end)
            )
            injection_plus_context.append(
                CaseInterval(
                    case_id,
                    min(context_start, int(audit_row["start_ms"])),
                    max(context_end, int(audit_row["end_ms"])),
                )
            )
        by_window[str(window_seconds)] = {
            "context_only": _group_summary(
                build_interval_overlap_groups(context_only)
            ),
            "injection_union_context": _group_summary(
                build_interval_overlap_groups(injection_plus_context)
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": index["dataset"],
        "case_count": len(inputs),
        "manifest_directory": str(manifest_directory),
        "manifest_sha256": _sha256(manifest_directory / "manifest.json"),
        "interval_semantics": "half-open [start_ms, end_ms)",
        "grouping_semantics": (
            "connected components of directly overlapping intervals; "
            "boundary-touching intervals do not overlap"
        ),
        "windows_seconds": by_window,
        "limitations": [
            "diagnostic only; it does not assign folds",
            "window selection additionally requires telemetry coverage evidence",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-manifest-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--windows-seconds",
        default=_parse_windows("30,60,300,600,1800"),
        type=_parse_windows,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose(args.gaia_manifest_dir, args.windows_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(result["windows_seconds"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
