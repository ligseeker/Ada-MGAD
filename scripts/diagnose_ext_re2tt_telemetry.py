#!/usr/bin/env python
"""E2: stream RE2-TT raw telemetry into a deterministic extension diagnostic.

Reuses ``src.data.telemetry_diagnostics.diagnose_re2ob`` unchanged; only the
dataset label differs. Writes ``artifacts/ext/re2tt/telemetry_diagnostics.json``
and never touches the frozen P1 diagnostic.

The report is strictly label-free: it summarizes coverage for all 68 candidate
services without knowing which one is the root. Root-conditioned coverage (for
example the ts-train-service log gap) is therefore recorded by the E5 gate audit,
which is allowed to read labels, not here.

``--windows-seconds`` defaults to the frozen P1 set plus 120 s, because the metric
onset candidates under audit are 30/60/120 s. Adding a diagnostic window does not
change any protocol constant.
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RE2TT_PROFILE, load_rcaeval_cases
from src.data.telemetry_diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    diagnose_re2ob,
)

from scripts.prepare_ext_re2tt_manifests import (
    DEFAULT_RAW_PATH,
    _assert_isolated,
)


DEFAULT_OUTPUT = "artifacts/ext/re2tt/telemetry_diagnostics.json"
EXTENSION_WINDOWS_SECONDS = (30, 60, 120, 300, 600, 1800)


def _positive_csv(value: str):
    parsed = tuple(sorted(set(int(item.strip()) for item in value.split(","))))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--windows-seconds", type=_positive_csv, default=EXTENSION_WINDOWS_SECONDS
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")

    output = Path(args.output)
    _assert_isolated(output.parent)

    adapter = load_rcaeval_cases(args.raw_path, RE2TT_PROFILE)
    report = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "evidence_scope": "raw telemetry streaming diagnostics",
        "extension": "re2tt_protocol_extension",
        "definitions": {
            "metric_missingness": (
                "empty/NaN value cells plus separately reported per-file "
                "scheduled timestamp gaps"
            ),
            "event_parse_failure": "missing required field or unparseable timestamp",
            "event_activity_coverage": (
                "occupied 30s bins; absence is not classified as missingness"
            ),
            "candidate_windows_seconds": list(args.windows_seconds),
        },
        "re2tt": diagnose_re2ob(
            args.raw_path,
            adapter,
            chunk_rows=args.chunk_rows,
            windows_seconds=args.windows_seconds,
            max_cases=args.max_cases,
            progress_every=max(1, args.progress_every),
            dataset=RE2TT_PROFILE.dataset,
            progress_label="RE2-TT cases",
        ),
        "limitations": [
            "no T_pre/T_post or split strategy is selected by this diagnostic",
            "full telemetry content checksums are not computed",
            "trace-derived graphs are identified as possible but not materialized",
            "root-conditioned coverage is deliberately absent; see the E5 gate audit",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    section = report["re2tt"]
    print(
        json.dumps(
            {
                "output": str(output),
                "scan_scope": section["scan_scope"],
                "cases": section["cases"],
                "rows": {
                    name: values["rows"]
                    for name, values in section["modalities"].items()
                },
                "incomplete_files": {
                    name: len(values["incomplete_files"])
                    for name, values in section["modalities"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
