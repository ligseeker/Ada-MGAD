"""P6-C1 G3 fold-local, prefix-fitted detector input materialization.

This module does not train a detector or create an experiment run. A caller
must choose a new fold directory; the writer refuses to replace any output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .gaia_preprocessing.materialize import (
    _raw_train_binding, _schema_payload, validate_transformed_modalities,
)
from .gaia_preprocessing.raw import (
    fit_logs, fit_metric, fit_trace, transform_logs,
    transform_metric_with_observability, transform_trace_with_diagnostics,
)
from .gaia_preprocessing.schema import load_frozen_preprocessing_schema
from .protocol import GAIA_SERVICES, ad_preprocessing_config_sha256


ROOT = Path(__file__).resolve().parents[2]
G2_CONFIG = ROOT / "configs/e2e/gaia_p6_c1_g2_v1_1.json"
G2_CONFIG_SHA256 = "a8b0d67f9aa06f4fe01fceaa982a9742145082be88096d73d310f4aa02a791ac"
SEGMENTS = ("fit", "selection", "generation")
EXPECTED_DIMENSIONS = {"raw_node": 48, "log_len": 32, "raw_edge": 8}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _bound_path(binding: Mapping[str, object]) -> Path:
    path = (ROOT / str(binding["path"])).resolve()
    if not _within(path, ROOT) or not path.is_file():
        raise ValueError("G2 bound input missing or outside repository: {}".format(path))
    if path.stat().st_size != int(binding["bytes"]) or _sha256(path) != binding["sha256"]:
        raise ValueError("G2 bound input drift: {}".format(path))
    return path


def verify_raw_content(manifest_path: Path, raw_root: Path) -> Mapping[str, object]:
    """Verify every historical raw byte before any fold output is written."""
    inventory = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw_root = Path(raw_root).resolve()
    if (inventory.get("schema_version") != "p6_c1_raw_content_inventory_v1"
            or inventory.get("raw_root") != str(raw_root)):
        raise ValueError("C1 raw content manifest identity drift")
    rows = inventory["files"]
    names = [row["path"] for row in rows]
    if (names != sorted(set(names)) or len(rows) != inventory["file_count"]
            or sum(row["bytes"] for row in rows) != inventory["total_bytes"]):
        raise ValueError("C1 raw content manifest records drift")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != inventory["records_sha256"]:
        raise ValueError("C1 raw content manifest digest drift")
    for row in rows:
        path = raw_root / row["path"]
        resolved = path.resolve()
        if (not _within(resolved, raw_root) or path.is_symlink()
                or not path.is_file()):
            raise ValueError("C1 raw file missing or outside raw root: {}".format(path))
        before = path.stat()
        if before.st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError("C1 raw file content drift: {}".format(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("C1 raw file changed while hashing: {}".format(path))
    return {"file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"],
            "records_sha256": inventory["records_sha256"],
            "manifest_sha256": _sha256(manifest_path)}


def _fold_intervals(protocol: Mapping[str, object], fold_number: int, grid_ms: int,
                    window_bins: int) -> Mapping[str, tuple]:
    if fold_number not in (1, 2, 3):
        raise ValueError("C1 has exactly three locked folds")
    fold = protocol["folds"][fold_number - 1]
    if fold["fold"] != fold_number:
        raise ValueError("C1 fold identity drift")
    intervals = fold["intervals_ms"]
    result = {name: tuple(int(value) for value in intervals[key]) for name, key in (
        ("fit", "detector_fit"), ("selection", "detector_selection"),
        ("generation", "anchor_generation"))}
    for name in SEGMENTS:
        start, end = result[name]
        if (start >= end or start % grid_ms or end % grid_ms
                or end - start <= window_bins * grid_ms):
            raise ValueError("C1 {} interval cannot produce segment-local windows".format(name))
    if result["fit"][1] != result["selection"][0] or result["selection"][1] != result["generation"][0]:
        raise ValueError("C1 fold segments must be adjacent and ordered")
    return result


def _source_hashes(config_path: Path, policy_path: Path, protocol_path: Path) -> Mapping[str, str]:
    paths = {
        "c1_fold_preprocessing": Path(__file__),
        "gaia_raw": ROOT / "src/e2e/gaia_preprocessing/raw.py",
        "gaia_materialize": ROOT / "src/e2e/gaia_preprocessing/materialize.py",
        "gaia_schema": ROOT / "src/e2e/gaia_preprocessing/schema.py",
        "gaia_metric": ROOT / "src/e2e/gaia_preprocessing/metric.py",
        "gaia_logs": ROOT / "src/e2e/gaia_preprocessing/logs.py",
        "gaia_traces": ROOT / "src/e2e/gaia_preprocessing/traces.py",
        "gaia_package": ROOT / "src/e2e/gaia_preprocessing/__init__.py",
        "parallel": ROOT / "src/e2e/parallel.py",
        "protocol": ROOT / "src/e2e/protocol.py",
        "drain_config": ROOT / "util/GAIA/gaia.ini",
        "base_config": config_path,
        "policy": policy_path,
        "g2_protocol": protocol_path,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _save_array(directory: Path, name: str, values: np.ndarray, staging: Path) -> Mapping[str, object]:
    path = directory / (name + ".npy")
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return {"path": str(path.relative_to(staging)), "shape": list(values.shape),
            "dtype": str(values.dtype), "bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def materialize_c1_fold(*, protocol_path: Path, fold_number: int, fold_root: Path,
                        runtime: Mapping[str, object]) -> Mapping[str, object]:
    """Fit once on the fold prefix and transform its three disjoint segments.

    The full raw-content manifest is checked before the staging directory is
    created. This is a fold-stage API, not a formal C1 all-stage CLI.
    """
    protocol_path = Path(protocol_path).resolve()
    if protocol_path != G2_CONFIG or _sha256(protocol_path) != G2_CONFIG_SHA256:
        raise ValueError("C1 G2 config path or bytes differ from the frozen v1.1 correction")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("protocol_id") != "P6-C1-G2-v1.1"
            or protocol.get("status") != "DESIGN_LOCKED_EXECUTION_BLOCKED_G3"):
        raise ValueError("C1 G2 protocol identity drift")
    fold_root = Path(fold_root).resolve()
    if (fold_root.name != "fold_{:02d}".format(fold_number)
            or fold_root.parent.name != "folds" or fold_root.exists()):
        raise ValueError("C1 fold output must be a new folds/fold_XX directory")
    base_path = _bound_path(protocol["bindings"]["base_config"])
    policy_path = _bound_path(protocol["bindings"]["preprocessing_policy"])
    raw_manifest_path = _bound_path(protocol["bindings"]["raw_content_manifest"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    raw_root = Path(base["gaia_raw_root"]).resolve()
    if _within(fold_root, raw_root):
        raise ValueError("C1 fold output cannot be inside raw input root")
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    history_seconds = int(protocol["detector"]["history_seconds"])
    if history_seconds * 1000 % grid_ms:
        raise ValueError("C1 history is not grid-aligned")
    window_bins = history_seconds * 1000 // grid_ms
    intervals = _fold_intervals(protocol, fold_number, grid_ms, window_bins)
    cpu_budget = int(protocol["resource_budget"]["ad_preprocessing_workers"])
    workers = int(runtime.get("workers", 1))
    if not 1 <= workers <= cpu_budget or runtime.get("start_method", "spawn") != "spawn":
        raise ValueError("C1 workers/start method differ from G2 budget")
    chunk_rows = int(runtime.get("chunk_rows", 100000))
    if chunk_rows < 1:
        raise ValueError("C1 chunk_rows must be positive")
    sources_before = _source_hashes(base_path, policy_path, protocol_path)
    raw_binding = verify_raw_content(raw_manifest_path, raw_root)
    if json.loads(raw_manifest_path.read_text(encoding="utf-8"))["config_sha256"] != _sha256(base_path):
        raise ValueError("C1 raw inventory base-config binding drift")
    if raw_binding["manifest_sha256"] != protocol["bindings"]["raw_content_manifest"]["sha256"]:
        raise ValueError("C1 raw content manifest binding drift")
    # Exclusive directory creation is the writer lock. An interrupted fold
    # remains visibly incomplete and is never adopted or replaced.
    staging = fold_root
    staging.mkdir(parents=True, exist_ok=False)
    _write_json_new(staging / "INCOMPLETE.json", {"status": "INCOMPLETE", "fold": fold_number})
    try:
        artifact_root = staging / "ad_artifacts"
        data_root = staging / "ad_data"
        artifact_root.mkdir()
        data_root.mkdir()
        fit_start, fit_end = intervals["fit"]
        metric_policy, log_policy, trace_policy = (policy[key] for key in ("metric", "logs", "traces"))
        metric_slots = int(metric_policy["base_slots"])
        metric_fit = fit_metric(
            raw_root / "metric/metric_split/metric", fit_start, fit_end,
            grid_ms=grid_ms, required_slots=metric_slots, max_slots=metric_slots,
            fill_max_intervals=int(metric_policy["fill_max_intervals"]),
            pearson_threshold=metric_policy["pearson_threshold"],
            spearman_threshold=metric_policy["spearman_threshold"],
            scope_quotas=metric_policy["scope_quotas"], workers=workers,
            start_method="spawn")
        log_fit = fit_logs(
            raw_root / "business/business_split/business", fit_start, fit_end,
            grid_ms=grid_ms, chunk_rows=chunk_rows,
            min_template_count=int(log_policy["min_template_count"]),
            min_template_bins=int(log_policy["min_template_bins"]),
            max_stable_templates=int(log_policy["stable_templates"]),
            config_path=ROOT / "util/GAIA/gaia.ini", workers=workers,
            start_method="spawn")
        trace_fit = fit_trace(
            raw_root / "trace/trace_split/trace", fit_start, fit_end,
            grid_ms=grid_ms, chunk_rows=chunk_rows,
            min_edge_rows=int(trace_policy["min_edge_rows"]),
            min_positive_bins=int(trace_policy["min_positive_bins"]),
            workers=workers, start_method="spawn")
        drain_path = artifact_root / "drain3_fit_state.json"
        _write_json_new(drain_path, {"schema_version": "gaia_ad_preprocessing_v2_drain_state",
                                     "state": log_fit.drain_state})
        schema_payload = _schema_payload(
            base_path, policy_path, _raw_train_binding(raw_root, fit_start, fit_end),
            ad_preprocessing_config_sha256(base), metric_fit, log_fit, trace_fit,
            drain_path)
        schema_payload["c1_fold"] = {"protocol_id": protocol["protocol_id"],
                                      "fold": fold_number,
                                      "fit_semantics": "fold_fit_prefix_is_schema_train",
                                      "intervals_ms": {name: list(intervals[name]) for name in SEGMENTS},
                                      "history_seconds": history_seconds}
        schema_payload["source_binding"]["g2_config_sha256"] = _sha256(protocol_path)
        schema_payload["source_binding"]["raw_content_manifest_sha256"] = raw_binding["manifest_sha256"]
        schema_payload["logs"]["drain_state_path"] = str(fold_root / "ad_artifacts/drain3_fit_state.json")
        schema_path = artifact_root / "schema.json"
        _write_json_new(schema_path, schema_payload)
        try:
            schema = load_frozen_preprocessing_schema(schema_path)
        except ValueError as exc:
            raise ValueError("PREFIX_SCHEMA_NO_GO: invalid prefix-fitted schema") from exc
        if dict(schema.dimensions) != EXPECTED_DIMENSIONS:
            raise ValueError("PREFIX_SCHEMA_NO_GO: fold cannot satisfy frozen 48/32/8 dimensions")
        graph = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float32)
        service_index = {service: index for index, service in enumerate(GAIA_SERVICES)}
        for source, destination in trace_fit.directed_edges:
            graph[service_index[source], service_index[destination]] = 1.0
            graph[service_index[destination], service_index[source]] = 1.0
        graph_record = _save_array(data_root, "graph", graph, staging)
        segment_records = {}
        for name in SEGMENTS:
            start, end = intervals[name]
            timestamps = np.arange(start, end, grid_ms, dtype=np.int64)
            metric = transform_metric_with_observability(
                metric_fit, raw_root / "metric/metric_split/metric", start, end,
                workers=workers, start_method="spawn")[0]
            logs = transform_logs(
                log_fit, raw_root / "business/business_split/business", start, end,
                chunk_rows=chunk_rows, workers=workers, start_method="spawn")
            trace, diagnostics = transform_trace_with_diagnostics(
                trace_fit, raw_root / "trace/trace_split/trace", start, end,
                chunk_rows=chunk_rows, workers=workers, start_method="spawn")
            validate_transformed_modalities(
                schema=schema, metric=metric, logs=logs, trace=trace,
                service_count=len(GAIA_SERVICES))
            if len(metric) != len(timestamps):
                raise ValueError("C1 transformed segment is not aligned to its locked grid")
            directory = data_root / name
            directory.mkdir()
            # Prediction time is the end of the target bin. A window ending
            # exactly at the half-open segment boundary belongs to the next
            # segment and must never enter this segment's detector loader.
            legal_indices = np.arange(len(timestamps) - window_bins, dtype=np.int64)
            files = {key: _save_array(directory, key, values, staging)
                     for key, values in (("timestamps", timestamps), ("metric", metric),
                                         ("log", logs), ("trace", trace),
                                         ("legal_window_indices", legal_indices))}
            segment_records[name] = {
                "interval_ms": [start, end], "time_bins": len(timestamps),
                "window_bins": window_bins, "legal_windows": len(legal_indices),
                "first_prediction_available_ms": start + window_bins * grid_ms,
                "last_prediction_available_ms": end - grid_ms,
                "trace_diagnostics": diagnostics, "files": files,
            }
        if verify_raw_content(raw_manifest_path, raw_root) != raw_binding:
            raise ValueError("C1 raw content changed during fold materialization")
        if _source_hashes(base_path, policy_path, protocol_path) != sources_before:
            raise ValueError("C1 preprocessing source changed during fold materialization")
        manifest = {
            "schema_version": "p6_c1_fold_ad_inputs_v1", "status": "COMPLETE",
            "protocol_id": protocol["protocol_id"], "fold": fold_number,
            "fold_root": str(fold_root), "raw_root": str(raw_root),
            "decision_inputs": ["fit"], "gt_labels_used_for_schema": False,
            "selection_or_generation_used_for_fit": False,
            "services": list(GAIA_SERVICES), "dimensions": dict(schema.dimensions),
            "raw_content": raw_binding, "source_sha256": sources_before,
            "schema": {"path": str(schema_path.relative_to(staging)), "sha256": _sha256(schema_path)},
            "drain_state": {"path": str(drain_path.relative_to(staging)), "sha256": _sha256(drain_path)},
            "graph": graph_record, "segments": segment_records,
            "workers": workers, "start_method": "spawn", "chunk_rows": chunk_rows,
        }
        _write_json_new(staging / "completion_manifest.json", manifest)
        (staging / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as exc:
        try:
            _write_json_new(staging / "failure.json", {"status": "INCOMPLETE",
                                                        "error_type": type(exc).__name__,
                                                        "message": str(exc)})
        except OSError:
            pass
        raise


def validate_c1_fold(fold_root: Path, *, protocol_path: Path) -> Mapping[str, object]:
    """Read-only seal check for a completed fold materialization."""
    fold_root = Path(fold_root).resolve()
    if Path(protocol_path).resolve() != G2_CONFIG or _sha256(G2_CONFIG) != G2_CONFIG_SHA256:
        raise ValueError("C1 G2 config path or bytes drift")
    if (fold_root / "INCOMPLETE.json").exists():
        raise ValueError("C1 fold output is incomplete")
    manifest = json.loads((fold_root / "completion_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_id") != "P6-C1-G2-v1.1":
        raise ValueError("C1 fold completion identity drift")
    if manifest.get("fold_root") != str(fold_root):
        raise ValueError("C1 fold output root drift")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if manifest["raw_content"]["manifest_sha256"] != protocol["bindings"]["raw_content_manifest"]["sha256"]:
        raise ValueError("C1 fold raw-content source drift")
    sources = manifest["source_sha256"]
    base_path = _bound_path(protocol["bindings"]["base_config"])
    policy_path = _bound_path(protocol["bindings"]["preprocessing_policy"])
    for name, digest in _source_hashes(
            base_path, policy_path, Path(protocol_path).resolve()).items():
        if sources.get(name) != digest:
            raise ValueError("C1 fold source drift: {}".format(name))
    records = [manifest["schema"], manifest["drain_state"], manifest["graph"]]
    for segment in SEGMENTS:
        records.extend(manifest["segments"][segment]["files"].values())
    for record in records:
        path = (fold_root / record["path"]).resolve()
        if not _within(path, fold_root) or not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError("C1 fold output drift: {}".format(record["path"]))
        if "bytes" in record and path.stat().st_size != record["bytes"]:
            raise ValueError("C1 fold output size drift: {}".format(record["path"]))
    schema = load_frozen_preprocessing_schema(fold_root / manifest["schema"]["path"])
    if dict(schema.dimensions) != EXPECTED_DIMENSIONS or schema.payload["c1_fold"]["fold"] != manifest["fold"]:
        raise ValueError("C1 fold schema identity drift")
    if Path(schema.payload["logs"]["drain_state_path"]).resolve() != (fold_root / manifest["drain_state"]["path"]).resolve():
        raise ValueError("C1 fold Drain3 state path drift")
    graph = np.load(fold_root / manifest["graph"]["path"], mmap_mode="r", allow_pickle=False)
    expected_graph = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float32)
    service_index = {service: index for index, service in enumerate(GAIA_SERVICES)}
    for source, destination in schema.payload["traces"]["directed_edges"]:
        expected_graph[service_index[source], service_index[destination]] = 1.0
        expected_graph[service_index[destination], service_index[source]] = 1.0
    if not np.array_equal(graph, expected_graph):
        raise ValueError("C1 fold graph differs from prefix-fitted schema")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if manifest.get("raw_root") != str(Path(base["gaia_raw_root"]).resolve()):
        raise ValueError("C1 fold raw input root drift")
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    window_bins = int(protocol["detector"]["history_seconds"]) * 1000 // grid_ms
    intervals = _fold_intervals(protocol, int(manifest["fold"]), grid_ms, window_bins)
    for name in SEGMENTS:
        segment = manifest["segments"][name]
        start, end = intervals[name]
        timestamps = np.load(fold_root / segment["files"]["timestamps"]["path"],
                             mmap_mode="r", allow_pickle=False)
        indices = np.load(fold_root / segment["files"]["legal_window_indices"]["path"],
                          mmap_mode="r", allow_pickle=False)
        if (segment["interval_ms"] != [start, end]
                or segment["time_bins"] != len(timestamps)
                or segment["window_bins"] != window_bins
                or segment["legal_windows"] != len(indices)
                or segment["first_prediction_available_ms"] != start + window_bins * grid_ms
                or segment["last_prediction_available_ms"] != end - grid_ms
                or not np.array_equal(timestamps, np.arange(start, end, grid_ms, dtype=np.int64))
                or not np.array_equal(indices, np.arange(len(timestamps) - window_bins, dtype=np.int64))):
            raise ValueError("C1 fold segment grid/window geometry drift: {}".format(name))
    return manifest
