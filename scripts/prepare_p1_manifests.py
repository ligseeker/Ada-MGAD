#!/usr/bin/env python
"""Generate deterministic, label-separated P1 manifests for GAIA and RE2-OB."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    build_overlap_groups,
    build_singleton_groups,
    load_gaia_cases,
    load_re2ob_cases,
    verify_manifest_bundle,
    write_manifest_bundle,
)


def _group_records(groups: Iterable[object]):
    return [
        {
            "case_id": group.case_id,
            "group_id": group.group_id,
            "group_size": group.group_size,
            "overlap_degree": group.overlap_degree,
        }
        for group in groups
    ]


def _group_summary(groups: Iterable[object]) -> Mapping[str, int]:
    values = tuple(groups)
    sizes_by_id: Dict[str, int] = {}
    for group in values:
        sizes_by_id[group.group_id] = group.group_size
    size_counts = Counter(sizes_by_id.values())
    return {
        "groups": len(sizes_by_id),
        "non_singleton_groups": sum(count for size, count in size_counts.items() if size > 1),
        "cases_in_non_singleton_groups": sum(
            size * count for size, count in size_counts.items() if size > 1
        ),
        "largest_group": max(sizes_by_id.values(), default=0),
    }


def prepare_gaia(raw_path: str, output_directory: Path) -> Mapping[str, object]:
    result = load_gaia_cases(raw_path)
    groups = build_overlap_groups(result.audit)
    sidecars = {
        "event_audit.jsonl": [
            {
                "case_id": row.case_id,
                "duration_seconds": row.duration_seconds,
                "end_ms": row.end_ms,
                "overlapping_case_ids": list(row.overlapping_case_ids),
                "source_index": row.source_index,
                "start_ms": row.start_ms,
            }
            for row in result.audit
        ],
        "excluded.jsonl": [
            {
                "observed_type": row.observed_type,
                "reason": row.reason,
                "source_index": row.source_index,
            }
            for row in result.excluded
        ],
        "groups.jsonl": _group_records(groups),
        "sources.jsonl": [
            {
                "logs_directory": result.source_layout.logs_directory,
                "metrics_directory": result.source_layout.metrics_directory,
                "run_table": result.source_layout.run_table,
                "source_id": "gaia-micross-2021-07-local",
                "traces_directory": result.source_layout.traces_directory,
            }
        ],
    }
    group_summary = _group_summary(groups)
    write_manifest_bundle(
        str(output_directory),
        dataset="GAIA-MicroSS-2021-07",
        inputs=result.inputs,
        labels=result.labels,
        trusted_sidecars=sidecars,
        metadata={
            "adapter": "src.data.gaia.load_gaia_cases",
            "excluded_records": len(result.excluded),
            "grouping": "connected components of directly overlapping injection intervals",
            "group_summary": group_summary,
            "status": "candidate cases; final context and split policy not frozen",
        },
    )
    verify_manifest_bundle(str(output_directory))
    return {
        "cases": len(result.inputs),
        "excluded": len(result.excluded),
        **group_summary,
    }


def prepare_re2ob(raw_path: str, output_directory: Path) -> Mapping[str, object]:
    result = load_re2ob_cases(raw_path)
    groups = build_singleton_groups(case.case_id for case in result.inputs)
    sidecars = {
        "excluded.jsonl": [
            {
                "reason": row.reason,
                "relative_directory": row.relative_directory,
            }
            for row in result.excluded
        ],
        "groups.jsonl": _group_records(groups),
        "sources.jsonl": [
            {
                "case_id": row.case_id,
                "inject_time_path": row.inject_time_path,
                "logs_path": row.logs_path,
                "metrics_path": row.metrics_path,
                "relative_directory": row.relative_directory,
                "traces_path": row.traces_path,
            }
            for row in result.sources
        ],
    }
    group_summary = _group_summary(groups)
    write_manifest_bundle(
        str(output_directory),
        dataset="RCAEval-RE2-OB",
        inputs=result.inputs,
        labels=result.labels,
        trusted_sidecars=sidecars,
        metadata={
            "adapter": "src.data.rcaeval.load_re2ob_cases",
            "excluded_cases": len(result.excluded),
            "grouping": "one official failure-case directory per group",
            "group_summary": group_summary,
            "status": "complete local case inventory; release identifier not pinned",
        },
    )
    verify_manifest_bundle(str(output_directory))
    return {
        "cases": len(result.inputs),
        "excluded": len(result.excluded),
        **group_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-path", required=True)
    parser.add_argument("--re2ob-path", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    summary = {
        "gaia": prepare_gaia(args.gaia_path, output_root / "gaia"),
        "re2ob": prepare_re2ob(args.re2ob_path, output_root / "re2ob"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
