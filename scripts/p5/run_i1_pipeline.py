#!/usr/bin/env python3
"""Orchestrate the frozen P5-I1 GAIA two-stage pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "full"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--raw-root", default=None,
                        help="Optional byte-layout-verified mirror of GAIA MicroSS.")
    parser.add_argument("--chunk-rows", default=500000, type=int)
    parser.add_argument("--gpu", default=True, type=lambda value: value.lower() == "true")
    return parser.parse_args()


def _run(arguments):
    command = [sys.executable] + list(arguments)
    print("RUN " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(PROJECT_ROOT))


def smoke(args):
    common = ["--config", args.config]
    _run(["scripts/p5/build_i1_protocol.py"] + common)
    _run(["scripts/p5/run_i1_ad.py", "smoke"] + common + ["--gpu", str(args.gpu).lower()])
    _run(["scripts/p5/run_i1_events.py", "smoke"] + common)
    _run(["scripts/p5/run_i1_rca_features.py", "smoke"] + common)
    _run(["scripts/p5/run_i1_rca.py", "smoke"] + common)
    _run(["scripts/p5/run_i1_e2e.py", "smoke"] + common)


def full(args):
    common = ["--config", args.config]
    raw = ["--raw-root", args.raw_root] if args.raw_root else []
    chunk = ["--chunk-rows", str(args.chunk_rows)]
    _run(["scripts/p5/build_i1_protocol.py"] + common)
    _run(["scripts/p5/run_i1_ad.py", "preprocess"] + common + raw + chunk)
    _run(["scripts/p5/run_i1_ad.py", "train"] + common + ["--gpu", str(args.gpu).lower()])
    _run(["scripts/p5/run_i1_events.py", "evaluate"] + common)
    _run(["scripts/p5/run_i1_rca_features.py", "index"] + common + raw + chunk)
    _run(["scripts/p5/run_i1_rca_features.py", "materialize"] + common)
    _run(["scripts/p5/run_i1_rca.py", "train-oracle"] + common)
    _run(["scripts/p5/run_i1_e2e.py", "evaluate"] + common)
    _run(["scripts/p5/finalize_i1_manifest.py"] + common)


def main():
    args = parse_args()
    if args.action == "smoke":
        smoke(args)
    else:
        full(args)
    print(json.dumps({
        "status": "COMPLETE",
        "action": args.action,
        "formal_result": args.action == "full",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
