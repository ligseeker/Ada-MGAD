#!/usr/bin/env python3
"""Capture immutable repository identities and historical-asset presence for P5."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import write_json  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_repository_state_v1"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _optional_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "none"


def _state(role: str, repo: Path) -> dict:
    return {
        "role": role,
        "repository": str(repo.resolve()),
        "origin": _git(repo, "remote", "get-url", "origin"),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "upstream": _optional_git(repo, "rev-parse", "--abbrev-ref", "@{upstream}"),
        "working_tree": _git(repo, "status", "--short") or "clean",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ada-mgad-main", required=True, type=Path)
    parser.add_argument("--p5-worktree", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--ada-rca-main", required=True, type=Path)
    parser.add_argument("--gaia-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    historical_assets = [
        "docs/README.md",
        "docs/PROJECT_CONTEXT.md",
        "docs/RESEARCH_STATUS.md",
        "docs/RCA_RESEARCH_DESIGN.md",
        "docs/BENCHMARK_PROTOCOL.md",
        "docs/DATASET_AUDIT.md",
        "docs/GAIA_INCLUSION_AUDIT.md",
        "docs/TELEMETRY_DIAGNOSTICS.md",
        "docs/SPLIT_DIAGNOSTICS.md",
        "docs/BASELINE_RESULTS.md",
        "docs/P2_METRIC_FEATURES.md",
        "docs/P2_MODALITY_SCHEMA_AUDIT.md",
        "docs/P2_C0_M_RESULTS.md",
        "docs/EXPERIMENT_LOG.md",
        "src/data/gaia.py",
        "src/data/event_purity.py",
        "src/data/split.py",
        "src/data/telemetry_diagnostics.py",
    ]
    run_table = args.gaia_root / "run/run/run/run_table_2021-07.csv"
    report = {
        "schema_version": SCHEMA_VERSION,
        "capture_note": "Working-tree values are the live state at capture time; initial state is also recorded in the report narrative.",
        "repositories": [
            _state("Ada-MGAD main", args.ada_mgad_main),
            _state("P5 audit worktree", args.p5_worktree),
            _state("Ada-MGAD rca-standalone historical", args.historical),
            _state("Ada-RCA canonical main", args.ada_rca_main),
        ],
        "historical_assets": {
            name: (args.historical / name).is_file() for name in historical_assets
        },
        "gaia": {
            "root": str(args.gaia_root.resolve()),
            "run_table": str(run_table.resolve()),
            "run_table_exists": run_table.is_file(),
        },
    }
    write_json(args.output, report)
    print("repositories={}; historical_assets_present={}/{}".format(
        len(report["repositories"]),
        sum(report["historical_assets"].values()),
        len(historical_assets),
    ))


if __name__ == "__main__":
    main()
