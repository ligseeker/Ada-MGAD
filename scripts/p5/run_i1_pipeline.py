#!/usr/bin/env python3
"""Orchestrate the frozen P5-I1 GAIA two-stage pipeline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_PATH = PROJECT_ROOT / "data/p5/i1/.pipeline.lock"


@contextmanager
def exclusive_pipeline_lock(path: Path = DEFAULT_LOCK_PATH):
    """Fail fast when another container is writing the canonical run outputs."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another P5-I1 pipeline owns the shared output lock: {}".format(lock_path)
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write("pid={}\n".format(os.getpid()))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "preprocess", "train-evaluate", "full"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--raw-root", default=None,
                        help="Optional byte-layout-verified mirror of GAIA MicroSS.")
    parser.add_argument("--chunk-rows", default=None, type=int)
    parser.add_argument("--raw-workers", default=None, type=int)
    parser.add_argument("--feature-workers", default=None, type=int)
    parser.add_argument("--event-workers", default=None, type=int)
    parser.add_argument("--case-chunk-size", default=None, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default=None)
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


def preprocess(args):
    common = ["--config", args.config]
    raw = ["--raw-root", args.raw_root] if args.raw_root else []
    chunk = ["--chunk-rows", str(args.chunk_rows)] if args.chunk_rows is not None else []
    method = ["--start-method", args.start_method] if args.start_method else []
    raw_workers = ["--workers", str(args.raw_workers)] if args.raw_workers is not None else []
    feature_workers = (
        ["--workers", str(args.feature_workers)] if args.feature_workers is not None else []
    )
    case_chunk = (
        ["--case-chunk-size", str(args.case_chunk_size)]
        if args.case_chunk_size is not None else []
    )
    _run(["scripts/p5/build_i1_protocol.py"] + common)
    _run(
        ["scripts/p5/run_i1_ad.py", "preprocess"]
        + common + raw + chunk + method + raw_workers
    )
    _run(
        ["scripts/p5/run_i1_rca_features.py", "index"]
        + common + raw + chunk + method + raw_workers
    )
    _run(
        ["scripts/p5/run_i1_rca_features.py", "materialize"]
        + common + method + feature_workers + case_chunk
    )


def train_evaluate(args):
    common = ["--config", args.config]
    _run(["scripts/p5/run_i1_ad.py", "train"] + common + ["--gpu", str(args.gpu).lower()])
    event_workers = (
        ["--workers", str(args.event_workers)] if args.event_workers is not None else []
    )
    event_method = ["--start-method", args.start_method] if args.start_method else []
    _run(["scripts/p5/run_i1_events.py", "evaluate"] + common + event_workers + event_method)
    _run(["scripts/p5/run_i1_rca.py", "train-oracle"] + common)
    _run(["scripts/p5/run_i1_e2e.py", "evaluate"] + common)
    _run(["scripts/p5/finalize_i1_manifest.py"] + common)


def main():
    args = parse_args()
    if args.action == "smoke":
        smoke(args)
    else:
        with exclusive_pipeline_lock():
            if args.action == "preprocess":
                preprocess(args)
            elif args.action == "train-evaluate":
                train_evaluate(args)
            else:
                preprocess(args)
                train_evaluate(args)
    print(json.dumps({
        "status": "COMPLETE",
        "action": args.action,
        "formal_result": args.action in ("train-evaluate", "full"),
        "formal_preprocessing": args.action in ("preprocess", "full"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
