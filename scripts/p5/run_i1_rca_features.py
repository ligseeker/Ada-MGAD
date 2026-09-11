#!/usr/bin/env python3
"""Build the GAIA raw RCA index and materialize frozen W300-B15 Z2 features."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex, IndexedSeries, build_raw_index
from src.e2e.parallel import atomic_save_npy, atomic_write_json, ordered_process_map
from src.e2e.protocol import (
    GAIA_SERVICES, load_config, preprocessing_runtime, sha256_file, write_json,
)
from src.e2e.rca_features import (
    TemporalSpec,
    extract_case_features_from_indicators,
    flatten_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("index", "materialize", "smoke", "all"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--index-root", default="data/p5/i1/rca_raw_index")
    parser.add_argument("--feature-root", default="data/p5/i1/rca_features")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument("--case-registry", default="artifacts/p5/i1/rca_case_registry.csv")
    parser.add_argument("--chunk-rows", default=None, type=int)
    parser.add_argument("--workers", default=None, type=int,
                        help="Processes for the selected action; defaults to frozen config.")
    parser.add_argument("--case-chunk-size", default=None, type=int)
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default=None)
    parser.add_argument("--limit-cases", default=0, type=int,
                        help="Smoke/debug limit only; zero is required for formal artifacts.")
    return parser.parse_args()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def temporal_spec(config):
    rca = config["rca"]
    return TemporalSpec(
        window_seconds=int(rca["window_seconds"]),
        bin_seconds=int(rca["bin_seconds"]),
        pre_bins=int(rca["pre_bins"]),
        post_bins=int(rca["post_bins"]),
        n_bins=int(rca["n_bins"]),
        onset_sentinel=float(rca["onset_sentinel_seconds"]),
        post_position_denominator=float(rca["post_position_denominator"]),
    )


def _health_from_arrays(features, counters):
    rows = np.asarray(features).reshape(-1, features.shape[-1])
    finite = np.isfinite(rows)
    constant = np.zeros(rows.shape[1], dtype=bool)
    finite_columns = np.all(finite, axis=0)
    constant[finite_columns] = np.ptp(rows[:, finite_columns], axis=0) <= 1e-12
    signatures = {
        __import__("hashlib").sha256(np.asarray(row, dtype="<f4").tobytes()).digest()
        for row in rows
    }
    candidate_rows = int(len(rows))
    return {
        "purpose": "implementation sanity check only; no representation selection",
        "cases": int(features.shape[0]),
        "candidate_rows": candidate_rows,
        "feature_dimension": int(features.shape[-1]),
        "finite_ratio": float(finite.mean()),
        "channel_available_ratio": float(counters["available_sum"] / counters["available_size"]),
        "mask_coverage": float(counters["mask_sum"] / counters["mask_size"]),
        "morphology_active_ratio": float(counters["active_sum"] / counters["active_size"]),
        "all_zero_rows": int(np.all(np.isclose(rows, 0.0), axis=1).sum()),
        "all_zero_row_ratio": float(np.all(np.isclose(rows, 0.0), axis=1).mean()),
        "all_masked_rows": int(counters["all_masked"]),
        "all_masked_row_ratio": float(counters["all_masked"] / candidate_rows),
        "constant_features": int(constant.sum()),
        "constant_feature_ratio": float(constant.mean()),
        "unique_candidate_signatures": int(len(signatures)),
        "candidate_signature_duplication": int(candidate_rows - len(signatures)),
        "candidate_signature_duplication_ratio": float(
            (candidate_rows - len(signatures)) / candidate_rows
        ),
    }


_MATERIALIZE_INDEX = None
_MATERIALIZE_SPEC = None


def _initialize_materializer(index_manifest_path, spec):
    global _MATERIALIZE_INDEX, _MATERIALIZE_SPEC
    _MATERIALIZE_INDEX = GaiaRcaRawIndex.from_manifest(Path(index_manifest_path))
    _MATERIALIZE_SPEC = spec


def _materialize_case_shard(task):
    if _MATERIALIZE_INDEX is None or _MATERIALIZE_SPEC is None:
        raise RuntimeError("RCA materialization worker was not initialized")
    start, stop, records, shard_path_value, metadata_path_value, binding = task
    shard_path = Path(shard_path_value)
    metadata_path = Path(metadata_path_value)
    expected_shape = (int(stop) - int(start), len(GAIA_SERVICES), 68)
    if shard_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = np.load(shard_path, mmap_mode="r")
        if (
            metadata.get("binding") != binding
            or tuple(cached.shape) != expected_shape
            or metadata.get("sha256") != sha256_file(shard_path)
        ):
            raise ValueError("RCA feature shard cache binding or checksum mismatch")
        return {
            "start": int(start), "stop": int(stop), "path": str(shard_path),
            "counters": metadata["counters"], "cache_hit": True,
        }
    values = np.empty(expected_shape, dtype=np.float32)
    counters = {
        "available_sum": 0, "available_size": 0,
        "mask_sum": 0, "mask_size": 0,
        "active_sum": 0, "active_size": 0,
        "all_masked": 0,
    }
    for offset, (case_id, anchor_ms) in enumerate(records):
        indicators = _MATERIALIZE_INDEX.case_indicators(int(anchor_ms), _MATERIALIZE_SPEC)
        value = extract_case_features_from_indicators(
            str(case_id), GAIA_SERVICES, indicators, _MATERIALIZE_SPEC
        )
        z2 = flatten_features(value, "z2")
        if z2.shape != (len(GAIA_SERVICES), 68) or not np.isfinite(z2).all():
            raise ValueError("invalid W300-B15 Z2 feature shape or value")
        values[offset] = z2.astype(np.float32)
        counters["available_sum"] += int(value.base[:, :, 7].sum())
        counters["available_size"] += int(value.base[:, :, 7].size)
        counters["mask_sum"] += int(value.q_mask.sum())
        counters["mask_size"] += int(value.q_mask.size)
        counters["active_sum"] += int(value.morphology_active.sum())
        counters["active_size"] += int(value.morphology_active.size)
        counters["all_masked"] += int(
            (~np.any(value.q_mask.astype(bool), axis=(1, 2))).sum()
        )
    atomic_save_npy(shard_path, values)
    atomic_write_json(metadata_path, {
        "binding": binding,
        "shape": list(values.shape),
        "sha256": sha256_file(shard_path),
        "counters": counters,
    })
    return {
        "start": int(start), "stop": int(stop), "path": str(shard_path),
        "counters": counters, "cache_hit": False,
    }


def materialize(
    config, index_root, feature_root, artifact_root, case_path, limit_cases=0,
    workers=1, case_chunk_size=128, start_method="spawn",
):
    import pandas as pd

    cases = pd.read_csv(
        case_path,
        usecols=lambda column: column in {"case_id", "start_ms", "split"},
    )
    required = {"case_id", "start_ms", "split"}
    if not required.issubset(cases.columns):
        raise ValueError("RCA case registry lacks prediction-visible columns")
    formal = int(limit_cases) == 0
    if limit_cases:
        cases = cases.iloc[:int(limit_cases)].copy()
    try:
        cases["start_ms"] = pd.to_numeric(cases["start_ms"], errors="raise").astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("RCA case registry start_ms must contain integer timestamps") from error
    if int(case_chunk_size) < 1:
        raise ValueError("RCA case chunk size must be at least one")
    feature_root.mkdir(parents=True, exist_ok=True)
    spec = temporal_spec(config)
    index_manifest_path = index_root / "index_manifest.json"
    index_manifest_sha = sha256_file(index_manifest_path)
    case_registry_sha = sha256_file(case_path)
    case_input_records = [
        {
            "case_id": str(row.case_id),
            "start_ms": int(row.start_ms),
            "split": str(row.split),
        }
        for row in cases.itertuples(index=False)
    ]
    case_input_sha = hashlib.sha256(
        json.dumps(
            case_input_records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    run_binding = {
        "schema": "p5_i1_rca_feature_shards_v1",
        "index_manifest_sha256": index_manifest_sha,
        "prediction_visible_case_inputs_sha256": case_input_sha,
        "limit_cases": int(limit_cases),
        "case_count": int(len(cases)),
        "case_chunk_size": int(case_chunk_size),
        "temporal_spec": {
            "window_seconds": spec.window_seconds,
            "bin_seconds": spec.bin_seconds,
            "pre_bins": spec.pre_bins,
            "post_bins": spec.post_bins,
            "n_bins": spec.n_bins,
            "onset_sentinel": spec.onset_sentinel,
            "post_position_denominator": spec.post_position_denominator,
        },
        "feature_dimension": int(config["rca"]["feature_dimension"]),
    }
    run_key = hashlib.sha256(
        json.dumps(run_binding, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    shard_root = feature_root / "shards" / run_key
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for start in range(0, len(cases), int(case_chunk_size)):
        stop = min(start + int(case_chunk_size), len(cases))
        records = tuple(
            (str(row.case_id), int(row.start_ms))
            for row in cases.iloc[start:stop].itertuples(index=False)
        )
        binding = dict(run_binding)
        binding.update({
            "start": start,
            "stop": stop,
            "records": [[case_id, anchor_ms] for case_id, anchor_ms in records],
        })
        stem = "{:06d}-{:06d}".format(start, stop)
        tasks.append((
            start, stop, records,
            str(shard_root / (stem + ".npy")),
            str(shard_root / (stem + ".json")),
            binding,
        ))
    phase_started = time.perf_counter()
    results, parallel_metadata = ordered_process_map(
        _materialize_case_shard, tasks,
        workers=workers, start_method=start_method,
        initializer=_initialize_materializer,
        initargs=(str(index_manifest_path), spec),
    )
    shard_wall_seconds = time.perf_counter() - phase_started
    feature_path = feature_root / "z2_features.npy"
    output_shape = (
        len(cases), len(GAIA_SERVICES), int(config["rca"]["feature_dimension"])
    )
    counters = {
        "available_sum": 0, "available_size": 0,
        "mask_sum": 0, "mask_size": 0,
        "active_sum": 0, "active_size": 0,
        "all_masked": 0,
    }
    merge_started = time.perf_counter()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=feature_path.name + ".", suffix=".tmp",
            dir=str(feature_root), delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        features = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32, shape=output_shape,
        )
        for result in results:
            features[int(result["start"]):int(result["stop"])] = np.load(
                result["path"], mmap_mode="r"
            )
            for key in counters:
                counters[key] += int(result["counters"][key])
        features.flush()
        del features
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, feature_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    features = np.load(feature_path, mmap_mode="r")
    merge_wall_seconds = time.perf_counter() - merge_started
    parallel_metadata = dict(parallel_metadata)
    parallel_metadata["phase_wall_seconds"] = {
        "case_shards": shard_wall_seconds,
        "deterministic_merge": merge_wall_seconds,
        "total": shard_wall_seconds + merge_wall_seconds,
    }
    health = _health_from_arrays(features, counters)
    health.update({
        "schema_version": "p5_i1_rca_feature_health_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "index_manifest_sha256": index_manifest_sha,
        "case_registry_sha256": case_registry_sha,
        "prediction_visible_case_inputs_sha256": case_input_sha,
        "parallel_execution": parallel_metadata,
        "shard_cache": {
            "run_key": run_key,
            "task_count": len(results),
            "cache_hits": sum(int(bool(result["cache_hit"])) for result in results),
            "policy": "binding and checksum validated; shard JSON is completion marker",
        },
    })
    health_name = "rca_feature_health.json" if formal else "rca_feature_smoke_health.json"
    atomic_write_json(artifact_root / health_name, health)
    case_ids_path = feature_root / "case_ids.npy"
    splits_path = feature_root / "splits.npy"
    anchors_path = feature_root / "anchors_ms.npy"
    atomic_save_npy(case_ids_path, cases["case_id"].astype(str).to_numpy(dtype="U32"))
    atomic_save_npy(splits_path, cases["split"].astype(str).to_numpy(dtype="U10"))
    atomic_save_npy(anchors_path, cases["start_ms"].to_numpy(dtype=np.int64))
    manifest = {
        "schema_version": "p5_i1_rca_feature_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "source_commit": config["source"]["ada_rca_commit"],
        "source_files": config["source"]["ada_rca_files"],
        "case_registry_sha256": case_registry_sha,
        "prediction_visible_case_inputs_sha256": case_input_sha,
        "difference_from_canonical": "temporal constants parameterized to W300-B15 only",
        "feature_variant": "z2",
        "shape": list(features.shape),
        "files": {
            "features": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
            "case_ids": {"path": str(case_ids_path.resolve()), "sha256": sha256_file(case_ids_path)},
            "splits": {"path": str(splits_path.resolve()), "sha256": sha256_file(splits_path)},
            "anchors_ms": {"path": str(anchors_path.resolve()), "sha256": sha256_file(anchors_path)},
        },
        "label_firewall": "feature construction received case_id/start_ms/split only; service and fault labels remain in the separate registry",
        "health_artifact": str((artifact_root / health_name).resolve()),
        "parallel_execution": parallel_metadata,
        "shard_cache": health["shard_cache"],
    }
    manifest_name = "rca_feature_manifest.json" if formal else "rca_feature_smoke_manifest.json"
    atomic_write_json(artifact_root / manifest_name, manifest)
    return manifest


def smoke(config, feature_root, artifact_root):
    spec = temporal_spec(config)
    anchor = 1_625_108_900_000
    starts = anchor - spec.window_seconds * 1000 + np.arange(spec.n_bins) * spec.bin_seconds * 1000
    metric = []
    logs = {}
    traces = {}
    for service_index, service in enumerate(GAIA_SERVICES):
        values = np.arange(spec.n_bins, dtype=float) + service_index
        values[spec.pre_bins:spec.pre_bins + 2] += 100.0
        metric.append(IndexedSeries(service, "smoke_metric", starts, values))
        logs[service] = (starts, np.mod(np.arange(spec.n_bins), 5).astype(np.uint8))
        traces[service] = [(starts, np.arange(spec.n_bins) % 7 == 0, values / 100.0)]
    index = GaiaRcaRawIndex(metric, logs, traces)
    feature_root.mkdir(parents=True, exist_ok=True)
    index_manifest = {
        "schema_version": "p5_i1_smoke_in_memory_index_v1",
        "metric_series": [], "logs": {}, "traces": {"parts": []},
    }
    write_json(feature_root / "index_manifest.json", index_manifest)
    indicators = index.case_indicators(anchor, spec)
    value = extract_case_features_from_indicators("smoke-case", GAIA_SERVICES, indicators, spec)
    z2 = flatten_features(value, "z2")
    health = {
        "schema_version": "p5_i1_rca_feature_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "formal_result": False,
        "fixture": "synthetic in-memory telemetry only",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "shape": list(z2.shape),
        "finite": bool(np.isfinite(z2).all()),
        "n_bins": spec.n_bins,
        "pre_bins": spec.pre_bins,
        "post_bins": spec.post_bins,
        "onset_sentinel": spec.onset_sentinel,
        "post_position_denominator": spec.post_position_denominator,
        "trace_error_rule": config["rca"]["trace_error_rule"],
    }
    if health["shape"] != [10, 68] or not health["finite"]:
        raise ValueError("RCA adapter smoke failed")
    write_json(artifact_root / "rca_feature_smoke_summary.json", health)
    return health


def main():
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    raw_root = Path(args.raw_root or config["gaia_raw_root"]).resolve()
    index_root = (PROJECT_ROOT / args.index_root).resolve()
    feature_root = (PROJECT_ROOT / args.feature_root).resolve()
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    case_path = (PROJECT_ROOT / args.case_registry).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    index_runtime = preprocessing_runtime(
        config, "rca_index", workers=args.workers, chunk_rows=args.chunk_rows,
        start_method=args.start_method,
    )
    feature_runtime = preprocessing_runtime(
        config, "rca_features", workers=args.workers, chunk_rows=args.chunk_rows,
        case_chunk_size=args.case_chunk_size, start_method=args.start_method,
    )
    result = None
    if args.action in ("index", "all"):
        timing = json.loads(
            (PROJECT_ROOT / "artifacts/p5/g0r2/raw_telemetry_timing.json").read_text()
        )
        result = build_raw_index(
            raw_root, index_root,
            timing["source_binding"]["current_inventory"],
            chunk_rows=index_runtime["chunk_rows"],
            workers=index_runtime["workers"],
            start_method=index_runtime["start_method"],
            provenance={
                "git_commit": git_head(),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "random_seed": int(config["random_seed"]),
            },
        )
    if args.action in ("materialize", "all"):
        result = materialize(
            config, index_root, feature_root, artifact_root, case_path,
            limit_cases=args.limit_cases,
            workers=feature_runtime["workers"],
            case_chunk_size=feature_runtime["case_chunk_size"],
            start_method=feature_runtime["start_method"],
        )
    if args.action == "smoke":
        result = smoke(config, feature_root, artifact_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
