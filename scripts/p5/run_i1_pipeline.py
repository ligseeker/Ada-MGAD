#!/usr/bin/env python3
"""Orchestrate shared GAIA V2 preprocessing and isolated experiment runs.

Training actions require a new independent ``--run-dir``; ``post-ad`` resumes
that same directory after its Ada-MGAD stage.  Smoke fixtures are never
promoted to formal results.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOCK_PATH = PROJECT_ROOT / "data/p5/v3_preprocessing_v2/.pipeline.lock"
V2_DATA_ROOT = "data/p5/v3_preprocessing_v2/ad"
V2_ARTIFACT_ROOT = "artifacts/p5/v3_preprocessing_v2/ad"
V2_CHECKPOINT_ROOT = "data/p5/v3_preprocessing_v2/checkpoint"


from src.e2e.experiment_layout import ExperimentLayout
from src.e2e.parallel import atomic_write_json
from src.e2e.protocol import load_config, sha256_file


@contextmanager
def exclusive_pipeline_lock(path: Path = DEFAULT_LOCK_PATH):
    """Fail fast when another process owns the requested pipeline target."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another pipeline process owns the shared output lock: {}".format(
                    lock_path
                )
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
    parser.add_argument(
        "action",
        choices=("smoke", "preprocess", "ad-train", "post-ad", "train-evaluate", "full"),
    )
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3_preprocessing_v2.json")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Independent experiment directory. Required for ad-train, post-ad, "
            "train-evaluate, and full; existing directories are never overwritten."
        ),
    )
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


def _run(arguments, log_paths=()):
    command = [sys.executable] + list(arguments)
    rendered = "RUN " + shlex.join(command)
    print(rendered, flush=True)
    unique_logs = []
    for value in log_paths or ():
        path = Path(value)
        if path not in unique_logs:
            unique_logs.append(path)
    if not unique_logs:
        subprocess.check_call(command, cwd=str(PROJECT_ROOT))
        return

    handles = []
    process = None
    try:
        for path in unique_logs:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            handle.write(rendered + "\n")
            handle.flush()
            handles.append(handle)
        environment = dict(os.environ)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            for handle in handles:
                handle.write(line)
                handle.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()


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


def _file_record(path):
    path = Path(path).resolve()
    value = {"path": str(path), "status": "COMPLETE" if path.is_file() else "MISSING"}
    if path.is_file():
        value.update({"bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return value


def _required_paths(paths, purpose):
    missing = [str(Path(path).resolve()) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("{} inputs are missing: {}".format(purpose, missing))


def _config_path(args):
    path = Path(args.config)
    return (PROJECT_ROOT / path if not path.is_absolute() else path).resolve()


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def _tracked_worktree_dirty():
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(PROJECT_ROOT), text=True,
    )
    return bool(output.strip())


def _shared_input_manifest(config, config_path, layout):
    event_registry = Path(str(config["event_registry"]["path"]))
    if not event_registry.is_absolute():
        event_registry = PROJECT_ROOT / event_registry
    schema = Path(str(config["ad_preprocessing"]["frozen_schema_path"]))
    if not schema.is_absolute():
        schema = PROJECT_ROOT / schema
    files = {
        "config": config_path,
        "ad_data_manifest": layout.ad_preprocess_artifact_root / "ad_data_manifest.json",
        "frozen_preprocessing_schema": schema,
        "protocol_manifest": layout.protocol_root / "protocol_manifest.json",
        "split_manifest": layout.protocol_root / "split_manifest.json",
        "gt_provenance": layout.protocol_root / "provenance.json",
        "event_registry": event_registry,
        "rca_index_manifest": layout.rca_index_root / "index_manifest.json",
        "rca_gt_feature_manifest": layout.rca_gt_artifact_root / "rca_feature_manifest.json",
        "rca_gt_case_registry": layout.rca_gt_case_registry,
        "rca_gt_features": layout.rca_gt_feature_root / "z2_features.npy",
        "rca_gt_case_ids": layout.rca_gt_feature_root / "case_ids.npy",
        "rca_gt_splits": layout.rca_gt_feature_root / "splits.npy",
        "rca_gt_anchors": layout.rca_gt_feature_root / "anchors_ms.npy",
    }
    return {
        "schema_version": "gaia_experiment_inputs_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(layout.run_dir),
        "shared_roots": {
            key: value for key, value in layout.as_dict().items()
            if key in {
                "ad_data_root", "ad_preprocess_artifact_root", "protocol_root",
                "rca_index_root", "rca_gt_feature_root", "rca_gt_artifact_root",
                "rca_gt_case_registry",
            }
        },
        "files": {name: _file_record(path) for name, path in files.items()},
    }


def _prepare_new_run(args, config, layout, require_post_inputs):
    ad_manifest = layout.ad_preprocess_artifact_root / "ad_data_manifest.json"
    _required_paths((layout.ad_data_root, ad_manifest), "Ada-MGAD")
    config_path = _config_path(args)
    input_manifest = _shared_input_manifest(config, config_path, layout)
    required_names = {"config", "ad_data_manifest", "frozen_preprocessing_schema"}
    if require_post_inputs:
        required_names.update({
            "protocol_manifest", "split_manifest", "gt_provenance",
            "event_registry", "rca_index_manifest", "rca_gt_feature_manifest",
            "rca_gt_case_registry", "rca_gt_features", "rca_gt_case_ids",
            "rca_gt_splits", "rca_gt_anchors",
        })
    incomplete = sorted(
        name for name in required_names
        if input_manifest["files"].get(name, {}).get("status") != "COMPLETE"
    )
    if incomplete:
        raise FileNotFoundError(
            "required shared inputs are not complete: {}".format(incomplete)
        )
    metadata = {
        "run_id": layout.run_dir.name,
        "action": args.action,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "tracked_worktree_dirty": _tracked_worktree_dirty(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "random_seed": int(config["random_seed"]),
        "command": list(sys.argv),
    }
    layout.create_new(metadata)
    atomic_write_json(
        layout.run_dir / "resolved_config.json",
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config": config,
            "experiment_layout": layout.as_dict(),
        },
    )
    atomic_write_json(
        layout.run_dir / "input_manifest.json",
        input_manifest,
    )


def _stage_logs(layout, filename):
    return (layout.logs_root / filename, layout.logs_root / "pipeline.log")


def _failure_details(stage, exc):
    return {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}


def _run_ad_stage(args, layout):
    layout.transition("INITIALIZED", "AD_RUNNING", {"stage": "ad-train", "pid": os.getpid()})
    try:
        _run([
            "scripts/p5/run_i1_ad.py", "train", "--config", args.config,
            "--gpu", str(args.gpu).lower(),
            "--data-root", str(layout.ad_data_root),
            "--preprocess-artifact-root", str(layout.ad_preprocess_artifact_root),
            "--artifact-root", str(layout.ad_artifact_root),
            "--checkpoint-dir", str(layout.ad_checkpoint_root),
        ], _stage_logs(layout, "01_ad_train.log"))
        _required_paths(
            (
                layout.ad_checkpoint_root / "best_train_f1.pt",
                layout.ad_checkpoint_root / "best_train_loss.pt",
                layout.ad_checkpoint_root / "last.pt",
                layout.ad_artifact_root / "ad_training_summary.json",
                layout.ad_artifact_root / "ad_train_predictions.csv",
                layout.ad_artifact_root / "ad_test_predictions.csv",
                layout.ad_artifact_root / "reconstruction_calibration.json",
            ),
            "completed Ada-MGAD",
        )
    except BaseException as exc:
        layout.transition("AD_RUNNING", "FAILED", _failure_details("ad-train", exc))
        raise
    layout.transition("AD_RUNNING", "AD_COMPLETE", {"stage": "ad-train"})


def _run_post_ad_stage(args, layout):
    _required_paths(
        (
            layout.ad_artifact_root / "ad_train_predictions.csv",
            layout.ad_artifact_root / "ad_test_predictions.csv",
            layout.ad_artifact_root / "ad_training_summary.json",
            layout.rca_index_root / "index_manifest.json",
            layout.rca_gt_case_registry,
            layout.rca_gt_feature_root / "z2_features.npy",
        ),
        "post-AD",
    )
    common = _common(args)
    method = _method(args)
    feature_workers = _workers(args.feature_workers)
    case_chunk = _case_chunk(args)
    event_workers = _workers(args.event_workers)
    layout.transition(
        "AD_COMPLETE", "POST_AD_RUNNING", {"stage": "post-ad", "pid": os.getpid()}
    )
    try:
        _run([
            "scripts/p5/run_i1_events.py", "evaluate",
            "--artifact-root", str(layout.event_root),
            "--train-predictions", str(layout.ad_artifact_root / "ad_train_predictions.csv"),
            "--test-predictions", str(layout.ad_artifact_root / "ad_test_predictions.csv"),
        ] + common + event_workers + method, _stage_logs(layout, "02_event_detection.log"))
        _run([
            "scripts/p5/run_i1_rca_features.py", "case-registry",
            "--anchor-mode", "detected",
            "--matching", str(layout.event_root / "event_matching.csv"),
            "--case-registry", str(layout.rca_detected_case_registry),
            "--index-root", str(layout.rca_index_root),
            "--feature-root", str(layout.rca_detected_feature_root),
            "--artifact-root", str(layout.rca_detected_artifact_root),
        ] + common, _stage_logs(layout, "03_rca_features.log"))
        _run([
            "scripts/p5/run_i1_rca_features.py", "materialize",
            "--index-root", str(layout.rca_index_root),
            "--case-registry", str(layout.rca_detected_case_registry),
            "--feature-root", str(layout.rca_detected_feature_root),
            "--artifact-root", str(layout.rca_detected_artifact_root),
        ] + common + method + feature_workers + case_chunk,
            _stage_logs(layout, "03_rca_features.log"))
        _run([
            "scripts/p5/run_i1_rca.py", "train-v3",
            "--feature-root", str(layout.rca_gt_feature_root),
            "--case-registry", str(layout.rca_gt_case_registry),
            "--detected-feature-root", str(layout.rca_detected_feature_root),
            "--detected-case-registry", str(layout.rca_detected_case_registry),
            "--model-path", str(layout.rca_model_path),
            "--artifact-root", str(layout.rca_artifact_root),
        ] + common, _stage_logs(layout, "04_rca_train.log"))
        _run([
            "scripts/p5/run_i1_e2e.py", "evaluate",
            "--artifact-root", str(layout.rca_artifact_root),
            "--index-manifest", str(layout.rca_index_root / "index_manifest.json"),
            "--model-path", str(layout.rca_model_path),
            "--detected-feature-path", str(layout.rca_detected_feature_root / "z2_features.npy"),
            "--case-registry", str(layout.rca_gt_case_registry),
            "--event-matching", str(layout.event_root / "event_matching.csv"),
            "--test-node-predictions", str(layout.ad_artifact_root / "ad_test_predictions.csv"),
            "--oracle-predictions", str(layout.rca_artifact_root / "rca_oracle_predictions.csv"),
            "--root-frequency-predictions", str(layout.rca_artifact_root / "root_frequency_predictions.csv"),
            "--detected-predictions", str(layout.rca_artifact_root / "rca_detected_predictions.csv"),
        ] + common, _stage_logs(layout, "05_e2e.log"))
        _run([
            "scripts/p5/finalize_v3_manifest.py", "--run-dir", str(layout.run_dir),
        ] + common, _stage_logs(layout, "05_e2e.log"))
        _required_paths(
            (layout.run_dir / "run_manifest.json", layout.run_dir / "final_report.md"),
            "finalized experiment",
        )
    except BaseException as exc:
        layout.transition("POST_AD_RUNNING", "FAILED", _failure_details("post-ad", exc))
        raise
    layout.transition("POST_AD_RUNNING", "COMPLETE", {"stage": "post-ad"})


def main():
    args = parse_args()
    if args.action == "smoke":
        smoke(args)
        state = None
    elif args.action == "preprocess":
        with exclusive_pipeline_lock():
            preprocess(args)
        state = None
    else:
        if not args.run_dir:
            raise SystemExit(
                "--run-dir is required for {} (for example: "
                "--run-dir experiments/gaia/<run-id>)".format(args.action)
            )
        config = load_config(_config_path(args))
        layout = ExperimentLayout.resolve(PROJECT_ROOT, config, Path(args.run_dir))

        if args.action == "post-ad":
            layout.load_existing()
            with exclusive_pipeline_lock(layout.lock_path):
                _run_post_ad_stage(args, layout)
        elif args.action == "ad-train":
            _prepare_new_run(args, config, layout, require_post_inputs=False)
            with exclusive_pipeline_lock(layout.lock_path):
                _run_ad_stage(args, layout)
        elif args.action == "train-evaluate":
            _prepare_new_run(args, config, layout, require_post_inputs=True)
            with exclusive_pipeline_lock(layout.lock_path):
                _run_ad_stage(args, layout)
                _run_post_ad_stage(args, layout)
        elif args.action == "full":
            # Refuse an existing run before touching the shared preprocessing
            # tree.  The run itself is created only after preprocessing has
            # completed and all reusable inputs can be validated.
            if layout.run_dir.exists():
                raise FileExistsError(
                    "run directory already exists: {}".format(layout.run_dir)
                )
            with exclusive_pipeline_lock():
                preprocess(args)
            _prepare_new_run(args, config, layout, require_post_inputs=True)
            with exclusive_pipeline_lock(layout.lock_path):
                _run_ad_stage(args, layout)
                _run_post_ad_stage(args, layout)
        else:  # pragma: no cover - argparse constrains this branch.
            raise ValueError("unsupported action: {}".format(args.action))
        state = layout.read_state()

    print(json.dumps({
        "status": "COMPLETE",
        "action": args.action,
        "run_dir": str(layout.run_dir) if state is not None else None,
        "run_state": state["status"] if state is not None else None,
        "formal_result": (
            state is not None and state["status"] == "COMPLETE"
        ),
        "full_gaia_preprocessing_executed": args.action in ("preprocess", "full"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
