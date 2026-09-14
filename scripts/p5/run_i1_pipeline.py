#!/usr/bin/env python3
"""Orchestrate the GAIA V2 preprocessing and downstream stages.

The ``preprocess`` and ``train-evaluate`` actions are operator-run actions.
This script does not turn smoke fixtures into formal results.
"""

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
DEFAULT_LOCK_PATH = PROJECT_ROOT / "data/p5/v3_preprocessing_v2/.pipeline.lock"
V2_DATA_ROOT = "data/p5/v3_preprocessing_v2/ad"
V2_ARTIFACT_ROOT = "artifacts/p5/v3_preprocessing_v2/ad"
V2_CHECKPOINT_ROOT = "data/p5/v3_preprocessing_v2/checkpoint"
V2_EVENT_ROOT = "artifacts/p5/v3_preprocessing_v2/events"


@contextmanager
def exclusive_pipeline_lock(path: Path = DEFAULT_LOCK_PATH):
    """Fail fast when another process owns the canonical V2 output tree."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another V2 pipeline owns the shared output lock: {}".format(lock_path)
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
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3_preprocessing_v2.json")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--chunk-rows", default=None, type=int)
    parser.add_argument("--raw-workers", default=None, type=int,
                        help="Workers for raw-index and preprocessing parse/map stages.")
    parser.add_argument("--ad-workers", default=None, type=int)
    parser.add_argument("--feature-workers", default=None, type=int)
    parser.add_argument("--event-workers", default=None, type=int)
    parser.add_argument("--case-chunk-size", default=None, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default=None)
    parser.add_argument("--gpu", default=False, type=lambda value: value.lower() == "true")
    return parser.parse_args()


def _run(arguments):
    command = [sys.executable] + list(arguments)
    print("RUN " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(PROJECT_ROOT))


def _common(args):
    return ["--config", args.config]


def _raw_root(args):
    return ["--raw-root", args.raw_root] if args.raw_root else []


def _chunk(args):
    return ["--chunk-rows", str(args.chunk_rows)] if args.chunk_rows is not None else []


def _method(args):
    return ["--start-method", args.start_method] if args.start_method else []


def _workers(value):
    return ["--workers", str(value)] if value is not None else []


def _case_chunk(args):
    return ["--case-chunk-size", str(args.case_chunk_size)] if args.case_chunk_size is not None else []


def smoke(args):
    common = _common(args)
    _run(["scripts/p5/run_i1_ad.py", "smoke"] + common + ["--gpu", str(args.gpu).lower()])
    _run(["scripts/p5/run_i1_events.py", "smoke"] + common)
    _run([
        "scripts/p5/run_i1_rca_features.py", "smoke", "--config", args.config,
        "--feature-root", "data/p5/v3/rca_features_smoke",
        "--artifact-root", "artifacts/p5/v3/rca_smoke",
    ])
    _run(["scripts/p5/run_i1_rca.py", "smoke"] + common)
    _run(["scripts/p5/run_i1_e2e.py", "smoke"] + common)


def preprocess(args):
    common = _common(args)
    raw = _raw_root(args)
    chunk = _chunk(args)
    method = _method(args)
    raw_workers = _workers(args.raw_workers)
    ad_workers = _workers(args.ad_workers if args.ad_workers is not None else args.raw_workers)
    feature_workers = _workers(args.feature_workers)
    case_chunk = _case_chunk(args)

    # Protocol/GT artifacts are regenerated against the V2 config. RCA feature
    # and model outputs below intentionally retain their historical V3 roots.
    _run(["scripts/p5/build_v3_gt.py"] + common + raw)
    _run(["scripts/p5/build_v3_protocol.py"] + common)
    _run(
        ["scripts/p5/run_i1_ad.py", "preprocess"]
        + common + raw + chunk + method + ad_workers
        + ["--data-root", V2_DATA_ROOT, "--artifact-root", V2_ARTIFACT_ROOT,
           "--checkpoint-dir", V2_CHECKPOINT_ROOT]
    )
    _run([
        "scripts/p5/run_i1_rca_features.py", "case-registry",
        "--anchor-mode", "gt", "--case-registry",
        "artifacts/p5/v3/rca/rca_case_registry_gt.csv",
    ] + common)
    _run(
        ["scripts/p5/run_i1_rca_features.py", "index"]
        + common + raw + chunk + method + raw_workers
    )
    _run([
        "scripts/p5/run_i1_rca_features.py", "materialize",
        "--case-registry", "artifacts/p5/v3/rca/rca_case_registry_gt.csv",
        "--feature-root", "data/p5/v3/rca_features_gt",
        "--artifact-root", "artifacts/p5/v3/rca_gt_features",
    ] + common + method + feature_workers + case_chunk)


def train_evaluate(args):
    common = _common(args)
    method = _method(args)
    feature_workers = _workers(args.feature_workers)
    case_chunk = _case_chunk(args)
    event_workers = _workers(args.event_workers)
    _run(["scripts/p5/run_i1_ad.py", "train"] + common + ["--gpu", str(args.gpu).lower(),
          "--data-root", V2_DATA_ROOT, "--artifact-root", V2_ARTIFACT_ROOT,
          "--checkpoint-dir", V2_CHECKPOINT_ROOT])
    _run([
        "scripts/p5/run_i1_events.py", "evaluate", "--artifact-root", V2_EVENT_ROOT,
        "--train-predictions", V2_ARTIFACT_ROOT + "/ad_train_predictions.csv",
        "--test-predictions", V2_ARTIFACT_ROOT + "/ad_test_predictions.csv",
    ] + common + event_workers + method)
    _run([
        "scripts/p5/run_i1_rca_features.py", "case-registry",
        "--anchor-mode", "detected",
        "--matching", V2_EVENT_ROOT + "/event_matching.csv",
        "--case-registry", "artifacts/p5/v3/rca/rca_case_registry_detected.csv",
    ] + common)
    _run([
        "scripts/p5/run_i1_rca_features.py", "materialize",
        "--case-registry", "artifacts/p5/v3/rca/rca_case_registry_detected.csv",
        "--feature-root", "data/p5/v3/rca_features_detected",
        "--artifact-root", "artifacts/p5/v3/rca_detected_features",
    ] + common + method + feature_workers + case_chunk)
    _run([
        "scripts/p5/run_i1_rca.py", "train-v3",
        "--feature-root", "data/p5/v3/rca_features_gt",
        "--case-registry", "artifacts/p5/v3/rca/rca_case_registry_gt.csv",
        "--detected-feature-root", "data/p5/v3/rca_features_detected",
        "--detected-case-registry", "artifacts/p5/v3/rca/rca_case_registry_detected.csv",
        "--artifact-root", "artifacts/p5/v3/rca",
    ] + common)
    _run([
        "scripts/p5/run_i1_e2e.py", "evaluate",
        "--artifact-root", "artifacts/p5/v3/rca",
        "--index-manifest", "data/p5/v3/rca_raw_index/index_manifest.json",
        "--model-path", "data/p5/v3/rca_model/conditional_logit.npz",
        "--case-registry", "artifacts/p5/v3/rca/rca_case_registry_gt.csv",
        "--event-matching", V2_EVENT_ROOT + "/event_matching.csv",
        "--test-node-predictions", V2_ARTIFACT_ROOT + "/ad_test_predictions.csv",
        "--oracle-predictions", "artifacts/p5/v3/rca/rca_oracle_predictions.csv",
        "--root-frequency-predictions", "artifacts/p5/v3/rca/root_frequency_predictions.csv",
        "--detected-predictions", "artifacts/p5/v3/rca/rca_detected_predictions.csv",
    ] + common)
    _run([
        "scripts/p5/finalize_v3_manifest.py",
        "--artifact-root", "artifacts/p5/v3",
        "--ad-artifact-root", V2_ARTIFACT_ROOT,
        "--ad-checkpoint-root", V2_CHECKPOINT_ROOT,
        "--event-artifact-root", V2_EVENT_ROOT,
    ] + common)


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
        "formal_result": args.action == "train-evaluate" or args.action == "full",
        "full_gaia_preprocessing_executed": args.action in ("preprocess", "full"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
