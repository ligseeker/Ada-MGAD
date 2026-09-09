#!/usr/bin/env python3
"""Create a deterministic hash manifest for the completed P5-G0R2 package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/p5/g0r2/audit_manifest.json")
    args = parser.parse_args()
    output_path = args.output.resolve()
    files = sorted(
        list((PROJECT_ROOT / "scripts/p5").glob("*.py"))
        + [PROJECT_ROOT / "docs/P5_GAIA_RECONCILIATION_AUDIT_V1.md"]
        + [
            path
            for path in (PROJECT_ROOT / "artifacts/p5/g0r2").iterdir()
            if path.is_file() and path.resolve() != output_path
        ],
        key=lambda path: str(path.relative_to(PROJECT_ROOT)),
    )
    run_table = Path("/home/zhangll24/RCA_project/datasets/GAIA/MicroSS/run/run/run/run_table_2021-07.csv")
    historical = Path("/home/zhangll24/RCA_project/Ada-MGAD-rca-standalone/artifacts/p1/telemetry_diagnostics.json")
    report = {
        "schema_version": "p5_g0r2_audit_manifest_v1",
        "scope": "audit scripts, report, and machine-readable evidence; no trained models",
        "inputs": {
            str(run_table): {"bytes": run_table.stat().st_size, "sha256": sha256_file(run_table)},
            str(historical): {"bytes": historical.stat().st_size, "sha256": sha256_file(historical)},
        },
        "files": {
            str(path.relative_to(PROJECT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "reproduction_entrypoints": [
            "scripts/p5/revalidate_gaia_events.py",
            "scripts/p5/audit_label_reconciliation.py",
            "scripts/p5/audit_injection_resolution.py",
            "scripts/p5/audit_raw_telemetry.py",
            "scripts/p5/audit_rca_compatibility.py",
            "scripts/p5/audit_p5_split_feasibility.py",
            "scripts/p5/build_historical_reuse_and_ledger.py",
            "scripts/p5/validate_p5_audit.py",
        ],
    }
    write_json(output_path, report)
    print("manifested {} files".format(len(files)))


if __name__ == "__main__":
    main()
