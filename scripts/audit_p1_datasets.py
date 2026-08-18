#!/usr/bin/env python
"""Summarize P1 adapter coverage without loading full telemetry payloads."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_gaia_cases, load_re2ob_cases


def _counts(values: Iterable[object]) -> Dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def build_audit(gaia_path: str, re2ob_path: str) -> Dict[str, object]:
    re2ob = load_re2ob_cases(re2ob_path)
    gaia = load_gaia_cases(gaia_path)

    overlap_cases = sum(bool(row.overlapping_case_ids) for row in gaia.audit)
    overlap_pairs = sum(len(row.overlapping_case_ids) for row in gaia.audit) // 2

    return {
        "schema_version": "p1_dataset_audit_v1",
        "evidence_scope": "adapter-and-event-inventory-only",
        "limitations": [
            "RCAEval release/checksum is not pinned yet",
            "telemetry missingness and relative-time coverage are not computed",
            "topology coverage is not computed",
            "GAIA overlap inclusion policy and context window are not frozen",
        ],
        "re2ob": {
            "source_path": str(Path(re2ob_path).resolve()),
            "valid_cases": len(re2ob.inputs),
            "excluded_cases": len(re2ob.excluded),
            "candidate_count_distribution": _counts(
                len(case_input.services) for case_input in re2ob.inputs
            ),
            "fault_type_distribution": _counts(
                label.fault_type for label in re2ob.labels
            ),
            "root_service_distribution": _counts(
                label.root_service for label in re2ob.labels
            ),
            "exclusions": [
                {
                    "relative_directory": row.relative_directory,
                    "reason": row.reason,
                }
                for row in re2ob.excluded
            ],
        },
        "gaia": {
            "source_path": str(Path(gaia_path).resolve()),
            "valid_cases": len(gaia.inputs),
            "excluded_records": len(gaia.excluded),
            "candidate_count_distribution": _counts(
                len(case_input.services) for case_input in gaia.inputs
            ),
            "fault_type_distribution": _counts(
                label.fault_type for label in gaia.labels
            ),
            "root_service_distribution": _counts(
                label.root_service for label in gaia.labels
            ),
            "exclusion_reason_distribution": _counts(
                row.reason for row in gaia.excluded
            ),
            "overlap_cases": overlap_cases,
            "overlap_pairs": overlap_pairs,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-path", required=True)
    parser.add_argument("--re2ob-path", required=True)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.gaia_path, args.re2ob_path)
    rendered = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
