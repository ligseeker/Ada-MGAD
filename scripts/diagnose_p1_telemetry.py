#!/usr/bin/env python
"""Stream raw GAIA and RE2-OB telemetry into a deterministic P1 diagnostic."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_gaia_cases, load_re2ob_cases
from src.data.telemetry_diagnostics import (
    DEFAULT_WINDOWS_SECONDS,
    build_telemetry_diagnostics,
)


def _positive_csv(value: str):
    parsed = tuple(sorted(set(int(item.strip()) for item in value.split(","))))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-path", required=True)
    parser.add_argument("--re2ob-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--windows-seconds",
        type=_positive_csv,
        default=DEFAULT_WINDOWS_SECONDS,
        help="Candidate symmetric windows used only for coverage diagnostics",
    )
    parser.add_argument(
        "--max-gaia-files-per-modality",
        type=int,
        help="Smoke-test limit; omit for a full GAIA scan",
    )
    parser.add_argument(
        "--max-re2ob-cases",
        type=int,
        help="Smoke-test limit; omit for all 90 official cases",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    if args.max_gaia_files_per_modality is not None and args.max_gaia_files_per_modality < 1:
        raise SystemExit("--max-gaia-files-per-modality must be positive")
    if args.max_re2ob_cases is not None and args.max_re2ob_cases < 1:
        raise SystemExit("--max-re2ob-cases must be positive")

    gaia = load_gaia_cases(args.gaia_path)
    re2ob = load_re2ob_cases(args.re2ob_path)
    report = build_telemetry_diagnostics(
        gaia_path=args.gaia_path,
        re2ob_path=args.re2ob_path,
        gaia_adapter=gaia,
        re2ob_adapter=re2ob,
        chunk_rows=args.chunk_rows,
        windows_seconds=args.windows_seconds,
        max_gaia_files_per_modality=args.max_gaia_files_per_modality,
        max_re2ob_cases=args.max_re2ob_cases,
        progress_every=args.progress_every,
    )

    output = Path(args.output)
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

    summary = {
        "output": str(output),
        "gaia": {
            "scan_scope": report["gaia"]["scan_scope"],
            "rows": {
                name: values["rows"]
                for name, values in report["gaia"]["modalities"].items()
            },
            "incomplete_files": {
                name: len(values["incomplete_files"])
                for name, values in report["gaia"]["modalities"].items()
            },
        },
        "re2ob": {
            "scan_scope": report["re2ob"]["scan_scope"],
            "cases": report["re2ob"]["cases"],
            "rows": {
                name: values["rows"]
                for name, values in report["re2ob"]["modalities"].items()
            },
            "incomplete_files": {
                name: len(values["incomplete_files"])
                for name, values in report["re2ob"]["modalities"].items()
            },
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
