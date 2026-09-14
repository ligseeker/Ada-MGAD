"""Single atomic GAIA Ada-MGAD V2 preprocessing orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np

from .raw import (
    fit_logs, fit_metric, fit_trace, transform_logs, transform_metric_with_observability,
    transform_trace,
)
from .schema import load_frozen_preprocessing_schema


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _raw_train_binding(raw_root: Path, start_ms: int, end_ms: int) -> str:
    digest = hashlib.sha256()
    for relative in ("metric/metric_split/metric", "business/business_split/business", "trace/trace_split/trace"):
        root = raw_root / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("*.csv")):
            digest.update(str(path.relative_to(raw_root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\n")
    digest.update("{}\0{}".format(start_ms, end_ms).encode("ascii"))
    return digest.hexdigest()


def _schema_payload(config_path, policy_path, raw_train_sha256, metric_fit, log_fit, trace_fit, drain_state_path):
    return {
        "schema_version": "gaia_ad_preprocessing_v2", "status": "FROZEN", "fit_split": "train",
        "decision_inputs": ["train"], "gt_labels_used": False, "test_used_for_selection": False,
        "source_binding": {"config_sha256": _sha256(config_path), "policy_sha256": _sha256(policy_path), "raw_train_sha256": raw_train_sha256},
        "metric": {
            "ordered_slots": list(metric_fit.slot_names) + ["global_observed_fraction", "host_applicable", "host_observed_fraction"],
            "slots": [{"name": slot.name, "scope": slot.scope, "logical_feature": slot.logical_feature,
                       "statistic": slot.statistic, "semantic_kind": slot.semantic_kind,
                       "targets": list(slot.targets), "source_keys": list(slot.source_keys)} for slot in metric_fit.slots],
            "scalers": {name: scaler.as_dict() for name, scaler in metric_fit.scalers.items()},
            "fill_max_age_ms": metric_fit.fill_max_age_ms,
        },
        "logs": {"stable_cluster_ids": list(log_fit.stable_cluster_ids), "ordered_slots": list(log_fit.slot_names),
                 "drain_state": log_fit.drain_state, "drain_state_path": str(drain_state_path),
                 "drain_state_sha256": _sha256(drain_state_path), "scalers": dict(log_fit.scalers),
                 "cluster_stats": dict(log_fit.cluster_stats)},
        "traces": {"status_order": ["200", "300", "400", "500"], "statistic_order": ["count", "mean_latency"],
                   "ordered_slots": ["{}_{}_log".format(status, statistic) for status in ("200", "300", "400", "500") for statistic in ("count", "mean_latency")],
                   "directed_edges": [list(edge) for edge in trace_fit.directed_edges],
                   "scalers": dict(trace_fit.scalers)},
    }


def validate_transformed_modalities(*, schema, metric: np.ndarray, logs: np.ndarray,
                                     trace: np.ndarray, service_count: int) -> Mapping[str, object]:
    if metric.ndim != 3 or logs.ndim != 3 or trace.ndim != 4:
        raise ValueError("modalities must have ranks 3/3/4")
    expected = {
        "metric": (metric.shape[0], service_count, schema.dimensions["raw_node"]),
        "logs": (metric.shape[0], service_count, schema.dimensions["log_len"]),
        "trace": (metric.shape[0], service_count, service_count, schema.dimensions["raw_edge"]),
    }
    actual = {"metric": metric.shape, "logs": logs.shape, "trace": trace.shape}
    if actual != expected:
        raise ValueError("trace shape or other transformed modality shape does not match frozen schema: {} != {}".format(actual, expected))
    for name, values in (("metric", metric), ("logs", logs), ("trace", trace)):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("{} contains non-finite or non-numeric values".format(name))
    return {"schema_sha256": schema.sha256, "dimensions": dict(schema.dimensions), "time_bins": int(metric.shape[0]), "finite": True}


def materialize_ad_inputs(*, schema_path: Path, config_path: Path, raw_root: Path,
                          data_root: Path, artifact_root: Path,
                          runtime: Mapping[str, object], policy_path: Path = None) -> Mapping[str, object]:
    from ..ad_data import build_registry_node_labels, build_semisupervised_mask, save_split_arrays
    from ..protocol import GAIA_SERVICES, assign_event_blocks, load_registry, temporal_blocks

    config_path, raw_root = Path(config_path).resolve(), Path(raw_root).resolve()
    data_root, artifact_root, schema_path = Path(data_root).resolve(), Path(artifact_root).resolve(), Path(schema_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_path = Path(policy_path or config["ad_preprocessing"]["policy_path"])
    if not policy_path.is_absolute():
        policy_path = config_path.parents[2] / policy_path
    if not policy_path.is_file():
        raise FileNotFoundError("V2 preprocessing policy: {}".format(policy_path))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    metric_policy = policy.get("metric", {})
    log_policy = policy.get("logs", {})
    trace_policy = policy.get("traces", {})
    blocks = temporal_blocks(config)
    if tuple(block.name for block in blocks) != ("train", "test"):
        raise ValueError("V2 preprocessing requires exactly train and test blocks")
    train, test = blocks
    grid_ms = int(config["ad"]["grid_seconds"]) * 1000
    chunk_rows = int(runtime.get("chunk_rows", 100000))
    workers = int(runtime.get("workers", 1))
    metric_workers = int(runtime.get("metric_workers", workers))
    log_workers = int(runtime.get("log_workers", workers))
    trace_workers = int(runtime.get("trace_workers", workers))
    cpu_budget = int(config["preprocessing"]["cpu_budget"])
    if min(metric_workers, log_workers, trace_workers) < 1 or max(metric_workers, log_workers, trace_workers) > cpu_budget:
        raise ValueError("modality workers must be between one and configured CPU budget {}".format(cpu_budget))
    start_method = str(runtime.get("start_method", "spawn"))
    raw_binding = _raw_train_binding(raw_root, train.start_ms, train.end_ms)

    metric_slots = int(metric_policy.get("base_slots", 45))
    metric_fit = fit_metric(raw_root / "metric/metric_split/metric", train.start_ms, train.end_ms,
                            grid_ms=grid_ms, required_slots=metric_slots, max_slots=metric_slots,
                            fill_max_intervals=int(metric_policy.get("fill_max_intervals", 2)),
                            pearson_threshold=metric_policy.get("pearson_threshold"),
                            spearman_threshold=metric_policy.get("spearman_threshold"),
                            scope_quotas=metric_policy.get("scope_quotas"),
                            workers=metric_workers, start_method=start_method)
    log_fit = fit_logs(raw_root / "business/business_split/business", train.start_ms, train.end_ms,
                       grid_ms=grid_ms, chunk_rows=chunk_rows,
                       min_template_count=int(log_policy.get("min_template_count", 2)),
                       min_template_bins=int(log_policy.get("min_template_bins", 2)),
                       max_stable_templates=int(log_policy.get("stable_templates", 17)),
                       config_path=config_path.parents[2] / "util/GAIA/gaia.ini",
                       workers=log_workers, start_method=start_method)
    trace_fit = fit_trace(raw_root / "trace/trace_split/trace", train.start_ms, train.end_ms,
                          grid_ms=grid_ms, chunk_rows=chunk_rows,
                          min_edge_rows=int(trace_policy.get("min_edge_rows", 1)),
                          min_positive_bins=int(trace_policy.get("min_positive_bins", 2)),
                          workers=trace_workers, start_method=start_method)
    drain_state_path = artifact_root / "drain3_train_state.json"
    _write_json_atomic(drain_state_path, {"schema_version": "gaia_ad_preprocessing_v2_drain_state", "state": log_fit.drain_state})
    if schema_path.exists():
        schema = load_frozen_preprocessing_schema(schema_path)
        binding = schema.payload["source_binding"]
        expected = {"config_sha256": _sha256(config_path), "policy_sha256": _sha256(policy_path), "raw_train_sha256": raw_binding}
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError("frozen V2 schema source binding differs from current sources")
    else:
        _write_json_atomic(schema_path, _schema_payload(config_path, policy_path, raw_binding, metric_fit, log_fit, trace_fit, drain_state_path))
        schema = load_frozen_preprocessing_schema(schema_path)
    expected_dimensions = {"raw_node": metric_slots + 3, "log_len": len(log_fit.slot_names), "raw_edge": 8}
    if schema.dimensions != expected_dimensions:
        raise ValueError("frozen schema dimensions do not match fitted V2 modality objects")
    if data_root.exists():
        raise FileExistsError("refusing to replace existing V2 output root: {}".format(data_root))
    staging = data_root.parent / (data_root.name + ".staging-{}".format(os.getpid()))
    if staging.exists():
        raise FileExistsError("staging output already exists: {}".format(staging))
    staging.mkdir(parents=True)
    try:
        metric_parts = {name: transform_metric_with_observability(metric_fit, raw_root / "metric/metric_split/metric", block.start_ms, block.end_ms, workers=metric_workers, start_method=start_method)[0]
                        for name, block in (("train", train), ("test", test))}
        log_parts = {"train": transform_logs(log_fit, raw_root / "business/business_split/business", train.start_ms, train.end_ms, chunk_rows=chunk_rows, workers=log_workers, start_method=start_method),
                     "test": transform_logs(log_fit, raw_root / "business/business_split/business", test.start_ms, test.end_ms, chunk_rows=chunk_rows, workers=log_workers, start_method=start_method)}
        trace_parts = {"train": transform_trace(trace_fit, raw_root / "trace/trace_split/trace", train.start_ms, train.end_ms, chunk_rows=chunk_rows, workers=trace_workers, start_method=start_method),
                       "test": transform_trace(trace_fit, raw_root / "trace/trace_split/trace", test.start_ms, test.end_ms, chunk_rows=chunk_rows, workers=trace_workers, start_method=start_method)}
        for split in ("train", "test"):
            validate_transformed_modalities(schema=schema, metric=metric_parts[split], logs=log_parts[split], trace=trace_parts[split], service_count=len(GAIA_SERVICES))
        assigned, purged = assign_event_blocks(load_registry(config, config_path.parents[2]), blocks)
        split_files, split_counts = {}, {}
        for block in blocks:
            timestamps = np.arange(block.start_ms, block.end_ms, grid_ms, dtype=np.int64)
            registry = assigned.loc[assigned["split"].astype(str) == block.name]
            labels = build_registry_node_labels(timestamps, registry, GAIA_SERVICES, int(config["ad"]["grid_seconds"]))
            mask = build_semisupervised_mask(labels, float(config["ad_model"]["label_percent"]), int(config["ad"]["window_bins"]))
            split_files[block.name] = save_split_arrays(staging, block.name, {"timestamps": timestamps, "metric": metric_parts[block.name], "log": log_parts[block.name], "trace": trace_parts[block.name], "labels": labels, "label_mask": mask})
            split_counts[block.name] = {"timestamps": len(timestamps), "windows": max(0, len(timestamps) - int(config["ad"]["window_bins"]) + 1)}
        graph = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float32)
        service_index = {service: index for index, service in enumerate(GAIA_SERVICES)}
        for source, destination in trace_fit.directed_edges:
            graph[service_index[source], service_index[destination]] = 1.0
            graph[service_index[destination], service_index[source]] = 1.0
        np.save(staging / "graph.npy", graph, allow_pickle=False)
        os.replace(str(staging), str(data_root))
        # save_split_arrays recorded the staging prefix; rebind every record
        # after the atomic directory rename and verify the final bytes.
        for split, records in split_files.items():
            for name, record in records.items():
                final_path = data_root / split / (name + ".npy")
                record["path"] = str(final_path.resolve())
                record["sha256"] = _sha256(final_path)
        manifest = {"schema_version": "gaia_ad_preprocessing_v2_manifest", "status": "COMPLETE", "formal_result": False,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(config_path.parents[2]), text=True).strip(),
                    "config_sha256": _sha256(config_path), "policy_sha256": _sha256(policy_path), "schema_sha256": _sha256(schema_path), "raw_train_sha256": raw_binding,
                    "decision_inputs": ["train"], "gt_labels_used_for_schema": False, "test_used_for_selection": False, "dimensions": dict(schema.dimensions), "schema_path": str(schema_path),
                    "services": list(GAIA_SERVICES), "split_counts": split_counts, "split_files": split_files,
                    "schema_artifacts": {"preprocessing": {"path": str(schema_path), "sha256": _sha256(schema_path)}, "drain_state": {"path": str(drain_state_path), "sha256": _sha256(drain_state_path)}},
                    "graph": {"path": str(data_root / "graph.npy"), "sha256": _sha256(data_root / "graph.npy"), "directed_edges": [list(edge) for edge in trace_fit.directed_edges]},
                    "metric": {"slots": list(metric_fit.slot_names)}, "logs": {"slots": list(log_fit.slot_names)}, "traces": {"directed_edges": [list(edge) for edge in trace_fit.directed_edges], "diagnostics": dict(trace_fit.diagnostics)}, "raw_injection_boundary_purged": len(purged), "workers": workers, "modality_workers": {"metric": metric_workers, "logs": log_workers, "traces": trace_workers}, "chunk_rows": chunk_rows, "start_method": start_method}
        _write_json_atomic(artifact_root / "ad_data_manifest.json", manifest)
        return manifest
    except Exception:
        if staging.exists():
            _write_json_atomic(staging / "INCOMPLETE.json", {"status": "INCOMPLETE"})
        raise
