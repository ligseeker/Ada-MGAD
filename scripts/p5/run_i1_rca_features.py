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

from src.e2e.gaia_rca_adapter import (
    GaiaRcaRawIndex, IndexedSeries, build_raw_index, validate_raw_index_manifest,
)
from src.e2e.parallel import atomic_save_npy, atomic_write_json, ordered_process_map
from src.e2e.protocol import (
    GAIA_SERVICES, assign_event_blocks, load_config, load_registry,
    preprocessing_runtime, purge_rca_cases, sha256_file, temporal_blocks,
    write_json,
)
from src.e2e.rca_features import (
    TemporalSpec,
    extract_case_features_from_indicators,
    flatten_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("case-registry", "index", "materialize", "smoke", "all"),
    )
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v3.json")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--index-root", default="data/p5/v3/rca_raw_index")
    parser.add_argument("--feature-root", default="data/p5/v3/rca_features_gt")
    parser.add_argument("--artifact-root", default="artifacts/p5/v3/rca")
    parser.add_argument("--case-registry", default="artifacts/p5/v3/rca/rca_case_registry_gt.csv")
    parser.add_argument("--anchor-mode", choices=("gt", "detected"), default="gt")
    parser.add_argument("--matching", default=None,
                        help="Event matching CSV for --anchor-mode detected.")
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
    finite_ratio = float(finite.mean()) if finite.size else 1.0
    available_ratio = (
        float(counters["available_sum"] / counters["available_size"])
        if counters["available_size"] else 0.0
    )
    mask_ratio = (
        float(counters["mask_sum"] / counters["mask_size"])
        if counters["mask_size"] else 0.0
    )
    active_ratio = (
        float(counters["active_sum"] / counters["active_size"])
        if counters["active_size"] else 0.0
    )
    return {
        "purpose": "implementation sanity check only; no representation selection",
        "cases": int(features.shape[0]),
        "candidate_rows": candidate_rows,
        "feature_dimension": int(features.shape[-1]),
        "finite_ratio": finite_ratio,
        "channel_available_ratio": available_ratio,
        "mask_coverage": mask_ratio,
        "morphology_active_ratio": active_ratio,
        "all_zero_rows": int(np.all(np.isclose(rows, 0.0), axis=1).sum()),
        "all_zero_row_ratio": float(np.all(np.isclose(rows, 0.0), axis=1).mean()) if candidate_rows else 0.0,
        "all_masked_rows": int(counters["all_masked"]),
        "all_masked_row_ratio": float(counters["all_masked"] / candidate_rows) if candidate_rows else 0.0,
        "constant_features": int(constant.sum()),
        "constant_feature_ratio": float(constant.mean()),
        "unique_candidate_signatures": int(len(signatures)),
        "candidate_signature_duplication": int(candidate_rows - len(signatures)),
        "candidate_signature_duplication_ratio": float(
            (candidate_rows - len(signatures)) / candidate_rows
        ) if candidate_rows else 0.0,
    }


_MATERIALIZE_INDEX = None
_MATERIALIZE_SPEC = None


def _initialize_materializer(index_manifest_path, spec):
    global _MATERIALIZE_INDEX, _MATERIALIZE_SPEC
    _MATERIALIZE_INDEX = GaiaRcaRawIndex.from_manifest(Path(index_manifest_path))
    _MATERIALIZE_SPEC = spec


def _is_v3(config):
    return str(config.get("schema_version", "")).startswith("p5_v3_")


def _config_path_for(config, config_path=None):
    """Return the config identity used in feature provenance.

    The public V1 helper is retained for historical tests, while every V3
    artifact is bound to the actual V3 JSON rather than a hard-coded legacy
    YAML path.
    """

    if config_path is not None:
        return Path(config_path).resolve()
    if _is_v3(config):
        return PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json"
    return PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"


def build_case_registry(
    config,
    output_path: Path,
    *,
    anchor_mode: str = "gt",
    matching_path: Path = None,
    config_path: Path = None,
):
    """Publish a label-separated RCA case registry for GT or detected anchors.

    The feature builder receives only ``case_id``, ``start_ms`` and ``split``.
    All service/fault columns remain in this registry for the scorer/evaluator,
    preserving the Ada-RCA label firewall.  Detected Test cases use the
    matched prediction availability time as their anchor; Train cases always
    use the GT start for model fitting.
    """

    import pandas as pd

    if not _is_v3(config):
        raise ValueError("case-registry generation is only available for V3")
    if anchor_mode not in ("gt", "detected"):
        raise ValueError("anchor_mode must be gt or detected")
    raw_registry = load_registry(config, PROJECT_ROOT)
    blocks = temporal_blocks(config)
    assigned, split_purged = assign_event_blocks(raw_registry, blocks)
    if anchor_mode == "gt":
        candidate = assigned.copy()
        candidate["anchor_type"] = "GT injection start"
        candidate["gt_start_ms"] = candidate["start_ms"].astype(np.int64)
        candidate["prediction_id"] = pd.NA
    else:
        if matching_path is None or not Path(matching_path).is_file():
            raise FileNotFoundError(
                "detected RCA case registry requires an event matching CSV"
            )
        matching = pd.read_csv(matching_path)
        required = {
            "match_status", "prediction_id", "case_id", "t_hat", "gt_start_ms",
            "gt_service", "fault_type", "split",
        }
        missing = sorted(required - set(matching.columns))
        if missing:
            raise ValueError("event matching lacks detected RCA columns: {}".format(missing))
        matched = matching.loc[
            (matching["match_status"].astype(str) == "matched")
            & (matching["split"].astype(str) == "test")
        ].copy()
        if matched["prediction_id"].isna().any() or matched["case_id"].isna().any():
            raise ValueError("matched Test events must have prediction and GT identities")
        test_block = next(block for block in blocks if block.name == "test")
        matched_times = pd.to_numeric(matched["t_hat"], errors="raise").astype(np.int64)
        if ((matched_times < test_block.start_ms) | (matched_times >= test_block.end_ms)).any():
            raise ValueError("detected Test anchors must lie inside the frozen Test timeline")
        assigned_test = assigned.loc[assigned["split"].astype(str) == "test"].copy()
        assigned_by_id = assigned_test.set_index(assigned_test["case_id"].astype(str), drop=False)
        unknown_case_ids = sorted(
            set(matched["case_id"].astype(str)) - set(assigned_by_id.index.astype(str))
        )
        if unknown_case_ids:
            raise ValueError(
                "detected matching refers to Test case IDs outside the V3 GT registry: {}".format(
                    unknown_case_ids[:3]
                )
            )
        matched["case_id"] = matched["case_id"].astype(str)
        for row in matched.itertuples(index=False):
            source = assigned_by_id.loc[str(row.case_id)]
            if str(row.gt_service) != str(source.service) or str(row.fault_type) != str(source.fault_type):
                raise ValueError("detected matching labels disagree with the frozen GT registry")
            if int(row.gt_start_ms) != int(source.start_ms):
                raise ValueError("detected matching onset disagrees with the frozen GT registry")
        if matched["prediction_id"].duplicated().any() or matched["case_id"].duplicated().any():
            raise ValueError("detected RCA matching must be one-to-one")
        train = assigned.loc[assigned["split"].astype(str) == "train"].copy()
        train["anchor_type"] = "GT injection start"
        train["gt_start_ms"] = train["start_ms"].astype(np.int64)
        train["prediction_id"] = pd.NA
        test = pd.DataFrame({
            "case_id": matched["case_id"].astype(str),
            "source_index": matched.get("source_index", pd.Series(index=matched.index, dtype="Int64")),
            "service": matched["gt_service"].astype(str),
            "labelled_service": matched["gt_service"].astype(str),
            "fault_type": matched["fault_type"].astype(str),
            "start_ms": pd.to_numeric(matched["t_hat"], errors="raise").astype(np.int64),
            "end_ms": pd.to_numeric(matched["t_hat"], errors="raise").astype(np.int64),
            "split": "test",
            "anchor_type": "detected prediction_available_time",
            "gt_start_ms": pd.to_numeric(matched["gt_start_ms"], errors="raise").astype(np.int64),
            "prediction_id": matched["prediction_id"].astype(str),
            "detection_delay_seconds": pd.to_numeric(
                matched.get("detection_delay_seconds", 0.0), errors="coerce"
            ),
        })
        candidate = pd.concat([train, test], ignore_index=True, sort=False)
    if candidate["case_id"].astype(str).duplicated().any():
        raise ValueError("RCA case registry contains duplicate case IDs")
    if candidate["start_ms"].isna().any():
        raise ValueError("RCA case registry contains missing anchors")
    candidate["case_id"] = candidate["case_id"].astype(str)
    candidate["start_ms"] = pd.to_numeric(candidate["start_ms"], errors="raise").astype(np.int64)
    # For detected Test anchors the context must be checked around t_hat, not
    # around the original GT start.  GT mode naturally has the same value.
    retained, rca_purged = purge_rca_cases(candidate, blocks, int(config["rca"]["window_seconds"]))
    retained = retained.sort_values(["start_ms", "split", "case_id"], kind="stable").reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retained.to_csv(output_path, index=False, lineterminator="\n")
    purge_path = output_path.with_name(output_path.stem + "_purged.csv")
    rca_purged.to_csv(purge_path, index=False, lineterminator="\n")
    metadata = {
        "schema_version": "p5_v3_rca_case_registry_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_INPUT_PENDING_FULL_RUN" if anchor_mode == "detected" else "FORMAL_INPUT",
        "formal_result": False,
        "anchor_mode": anchor_mode,
        "anchor_definition": (
            "Train GT start and matched Test prediction_available_time"
            if anchor_mode == "detected" else "GT injection start"
        ),
        "config_sha256": sha256_file(_config_path_for(config, config_path)),
        "event_registry_sha256": sha256_file(
            (PROJECT_ROOT / str(config["event_registry"]["path"])).resolve()
        ),
        "matching": (
            {"path": str(Path(matching_path).resolve()), "sha256": sha256_file(matching_path)}
            if matching_path is not None else None
        ),
        "input_rows": int(len(candidate)),
        "retained_rows": int(len(retained)),
        "split_counts": {
            name: int((retained["split"].astype(str) == name).sum())
            for name in ("train", "test")
        },
        "split_crossing_purged": int(len(split_purged)),
        "rca_context_purged": int(len(rca_purged)),
        "rca_context_purge_rule": "[anchor-300s,anchor+300s) must remain in the same Train/Test block",
        "files": {
            "registry": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
            "purged": {"path": str(purge_path.resolve()), "sha256": sha256_file(purge_path)},
        },
    }
    metadata_path = output_path.with_suffix(".json")
    write_json(metadata_path, metadata)
    return metadata


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
    workers=1, case_chunk_size=128, start_method="spawn", config_path=None,
    formal_result=None,
):
    import pandas as pd

    cases = pd.read_csv(
        case_path,
        usecols=lambda column: column in {"case_id", "start_ms", "split"},
    )
    required = {"case_id", "start_ms", "split"}
    if not required.issubset(cases.columns):
        raise ValueError("RCA case registry lacks prediction-visible columns")
    if _is_v3(config):
        split_values = set(cases["split"].astype(str))
        if "validation" in split_values:
            raise ValueError("V3 RCA feature materialization cannot consume Validation cases")
        if not split_values.issubset({"train", "test"}):
            raise ValueError("V3 RCA feature materialization requires Train/Test splits")
    formal = int(limit_cases) == 0 if formal_result is None else bool(formal_result)
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
    validate_raw_index_manifest(index_manifest_path)
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
        "schema": "p5_v3_rca_feature_shards_v1" if _is_v3(config) else "p5_i1_rca_feature_shards_v1",
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
        "schema_version": "p5_v3_rca_feature_health_v1" if _is_v3(config) else "p5_i1_rca_feature_health_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(_config_path_for(config, config_path)),
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
        "schema_version": "p5_v3_rca_feature_manifest_v1" if _is_v3(config) else "p5_i1_rca_feature_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(_config_path_for(config, config_path)),
        "source_commit": config["source"]["ada_rca_commit"],
        "source_files": config["source"]["ada_rca_files"],
        "case_registry_sha256": case_registry_sha,
        "prediction_visible_case_inputs_sha256": case_input_sha,
        "difference_from_canonical": (
            "raw GAIA adapter; temporal constants parameterized to frozen W300-B15"
            if _is_v3(config) else "temporal constants parameterized to W300-B15 only"
        ),
        "feature_variant": "z2",
        "shape": list(features.shape),
        "files": {
            "features": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
            "case_ids": {"path": str(case_ids_path.resolve()), "sha256": sha256_file(case_ids_path)},
            "splits": {"path": str(splits_path.resolve()), "sha256": sha256_file(splits_path)},
            "anchors_ms": {"path": str(anchors_path.resolve()), "sha256": sha256_file(anchors_path)},
        },
        "label_firewall": "feature construction received case_id/start_ms/split only; service and fault labels remain in the separate registry",
        "anchor_semantics": (
            "case registry start_ms is GT start for oracle or matched prediction_available_time for detected"
            if _is_v3(config) else "case registry start_ms is the V1 case anchor"
        ),
        "health_artifact": str((artifact_root / health_name).resolve()),
        "parallel_execution": parallel_metadata,
        "shard_cache": health["shard_cache"],
    }
    manifest_name = "rca_feature_manifest.json" if formal else "rca_feature_smoke_manifest.json"
    atomic_write_json(artifact_root / manifest_name, manifest)
    return manifest


def smoke(config, feature_root, artifact_root):
    if not _is_v3(config):
        raise ValueError("RCA feature smoke requires the V3 protocol config")
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
        "schema_version": "p5_v3_smoke_in_memory_index_v1",
        "metric_series": [], "logs": {}, "traces": {"parts": []},
    }
    write_json(feature_root / "index_manifest.json", index_manifest)
    indicators = index.case_indicators(anchor, spec)
    value = extract_case_features_from_indicators("smoke-case", GAIA_SERVICES, indicators, spec)
    z2 = flatten_features(value, "z2")
    health = {
        "schema_version": "p5_v3_rca_feature_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "formal_result": False,
        "fixture": "synthetic in-memory telemetry only",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(_config_path_for(config)),
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
    if args.action == "case-registry":
        matching_path = (
            (PROJECT_ROOT / args.matching).resolve() if args.matching else None
        )
        result = build_case_registry(
            config, case_path, anchor_mode=args.anchor_mode,
            matching_path=matching_path, config_path=config_path,
        )
    if args.action in ("index", "all"):
        run_table = raw_root / "run/run/run/run_table_2021-07.csv"
        if _is_v3(config):
            if not run_table.is_file():
                raise FileNotFoundError(run_table)
            if sha256_file(run_table) != str(config["run_table"]["sha256"]):
                raise ValueError("RCA raw index run table differs from the V3 provenance binding")
        result = build_raw_index(
            raw_root, index_root,
            chunk_rows=index_runtime["chunk_rows"],
            workers=index_runtime["workers"],
            start_method=index_runtime["start_method"],
            provenance={
                "git_commit": git_head(),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "random_seed": int(config["random_seed"]),
                "run_table_sha256": sha256_file(run_table) if run_table.is_file() else None,
                "protocol_id": str(config.get("protocol_id", "legacy")),
                "raw_adapter": "independent exact-timestamp telemetry; no Ada-MGAD tensors",
            },
        )
    if args.action in ("materialize", "all"):
        result = materialize(
            config, index_root, feature_root, artifact_root, case_path,
            limit_cases=args.limit_cases,
            workers=feature_runtime["workers"],
            case_chunk_size=feature_runtime["case_chunk_size"],
            start_method=feature_runtime["start_method"],
            config_path=config_path,
        )
    if args.action == "smoke":
        result = smoke(config, feature_root, artifact_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
