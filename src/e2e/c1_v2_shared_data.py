"""P6-C1-v2 shared-preprocessed fold slicing, sealing and validation.

This adapter never fits anything. It reads the frozen shared Train arrays, slices
them by the locked fold timestamps, creates legal 300 s detector windows, and
seals the result with source and output hashes. It cannot re-fit the Metric
schema, the scaler/statistics, Drain3 or the graph: those bytes are bound by the
protocol and only copied or sliced.

Evidence grade: supervision-OOS and detector-parameter-OOS only. The shared
preprocessing was fitted once on the original 70 % Train period with label-free
decisions; it may have observed label-free Train telemetry later than a fold
Generation interval. This adapter must never be described as
prefix-preprocessing-OOS or full-pipeline forward-OOS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .protocol import GAIA_SERVICES


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e2e/gaia_p6_c1_v2_supervision_oos.json"
CONFIG_SHA256 = "4e4529db1049fa5c04e658193b6b4e91eef0d9ad9e5d0efcd785dc776c475874"
PROTOCOL_ID = "P6-C1-SUPERVISION-OOS-v1"
RUN_ID = "c1-supervision-oos-v1-seed42"
SEGMENTS = ("fit", "selection", "generation")
SEGMENT_KEYS = {"fit": "detector_fit", "selection": "detector_selection",
                "generation": "anchor_generation"}
EXPECTED_DIMENSIONS = {"raw_node": 48, "log_len": 32, "raw_edge": 8}
ARRAY_NAMES = ("timestamps", "metric", "log", "trace")
SEGMENT_FILES = ARRAY_NAMES + ("legal_window_indices",)
FOLD_MANIFEST_SCHEMA = "p6_c1_v2_shared_fold_inputs_v1"
G1_MANIFEST_SCHEMA = "p6_c1_v2_g1_completion_v1"

ARRAY_BINDING_NAMES = ("metric", "log", "trace", "train_timestamps", "graph")
IDENTITY_BINDINGS = {
    "metric": "shared_train_metric",
    "log": "shared_train_log",
    "trace": "shared_train_trace",
    "train_timestamps": "shared_train_timestamps",
    "graph": "shared_ad_graph",
    "schema": "shared_frozen_schema",
    "ad_manifest": "shared_ad_manifest",
    "test_timestamps": "shared_test_timestamps",
}

# Every execution source is bound into the run lock; any drift fails a stage.
SOURCE_PATHS = (
    "scripts/p6/run_c1_v2.py",
    "scripts/p6/audit_c1_v2_g1_feasibility.py",
    "src/e2e/c1_v2_shared_data.py",
    "src/e2e/c1_v2_detector.py",
    "src/e2e/c1_v2_c2.py",
    "src/e2e/c1_oos_matching.py",
    "src/e2e/c1_common_cohort.py",
    "src/e2e/c1_shared_rca.py",
    "src/e2e/c1_test_scoring.py",
    "src/e2e/c1_evaluation.py",
    "src/e2e/rca_model.py",
    "src/e2e/rca_features.py",
    "src/e2e/gaia_rca_adapter.py",
    "src/e2e/system_trigger.py",
    "src/e2e/system_trigger_model.py",
    "src/e2e/system_trigger_data.py",
    "src/e2e/event_detection.py",
    "src/e2e/protocol.py",
    "src/e2e/ad_data.py",
    "src/model_util.py",
    "util/util.py",
    "util/GAIA/gaia.ini",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


def within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def bound_path(binding: Mapping[str, object]) -> Path:
    path = (ROOT / str(binding["path"])).resolve()
    if not within(path, ROOT) or not path.is_file():
        raise ValueError("P6-C1-v2 bound input missing or outside repository: {}".format(path))
    if path.stat().st_size != int(binding["bytes"]) or sha256(path) != str(binding["sha256"]):
        raise ValueError("P6-C1-v2 bound input drift: {}".format(path))
    return path


def record(path: Path, *, relative_to: Optional[Path] = None) -> Mapping[str, object]:
    path = Path(path)
    base = Path(relative_to) if relative_to is not None else ROOT
    return {"path": str(path.resolve().relative_to(Path(base).resolve())),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_protocol() -> Mapping[str, object]:
    """Load the frozen lock and fail closed on any byte drift."""
    if sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("P6-C1-v2 protocol config drift")
    protocol = read_json(CONFIG)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("run_id") != RUN_ID:
        raise ValueError("P6-C1-v2 protocol identity drift")
    if protocol.get("status") not in ("DESIGN_LOCKED_G1_STATIC_AUDIT_REQUIRED", "DESIGN_LOCKED"):
        raise ValueError("P6-C1-v2 protocol status drift")
    return protocol


def source_hashes() -> Mapping[str, str]:
    """Digest every bound execution source; a missing source fails closed."""
    hashes = {}
    for name in SOURCE_PATHS:
        path = ROOT / name
        if not path.is_file():
            raise ValueError("P6-C1-v2 bound source is missing: {}".format(name))
        hashes[name] = sha256(path)
    return hashes


def run_root(protocol: Mapping[str, object]) -> Path:
    root = (ROOT / str(protocol["output_root"])).resolve()
    if root != (ROOT / "experiments/p6/c1_supervision_oos" / RUN_ID).resolve():
        raise ValueError("P6-C1-v2 output root differs from the frozen run id")
    return root


def fold_intervals(protocol: Mapping[str, object], fold_number: int, *, grid_ms: int,
                   window_bins: int) -> Mapping[str, tuple]:
    if fold_number not in (1, 2, 3):
        raise ValueError("P6-C1-v2 has exactly three locked folds")
    fold = protocol["folds"][fold_number - 1]
    if int(fold["fold"]) != fold_number:
        raise ValueError("P6-C1-v2 fold identity drift")
    intervals = {}
    for name in SEGMENTS:
        start, end = (int(value) for value in fold["intervals_ms"][SEGMENT_KEYS[name]])
        if (start >= end or start % grid_ms or end % grid_ms
                or end - start <= window_bins * grid_ms):
            raise ValueError("P6-C1-v2 {} interval cannot host legal windows".format(name))
        intervals[name] = (start, end)
    if (intervals["fit"][1] != intervals["selection"][0]
            or intervals["selection"][1] != intervals["generation"][0]):
        raise ValueError("P6-C1-v2 fold segments must be adjacent and ordered")
    return intervals


def grid_geometry(protocol: Mapping[str, object], base: Mapping[str, object]):
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    history_ms = int(protocol["detector"]["history_seconds"]) * 1000
    if history_ms % grid_ms or history_ms // grid_ms != int(protocol["detector"]["window_bins"]):
        raise ValueError("P6-C1-v2 history/window geometry drift")
    return grid_ms, history_ms // grid_ms


def validate_g1_stage(protocol: Mapping[str, object]) -> Mapping[str, object]:
    """Read-only seal check of the run's static feasibility stage."""
    directory = run_root(protocol) / "g1_static"
    if (directory / "INCOMPLETE.json").exists():
        raise ValueError("P6-C1-v2 G1 stage is incomplete")
    manifest = read_json(directory / "completion_manifest.json")
    if (manifest.get("schema_version") != G1_MANIFEST_SCHEMA
            or manifest.get("status") != "COMPLETE"
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("run_id") != RUN_ID
            or manifest.get("config_sha256") != sha256(CONFIG)):
        raise ValueError("P6-C1-v2 G1 completion identity drift")
    for name, entry in manifest["files"].items():
        path = directory / name
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]):
            raise ValueError("P6-C1-v2 G1 output drift: {}".format(name))
    return manifest


def _display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _shared_identity(protocol: Mapping[str, object]) -> Mapping[str, object]:
    bindings = protocol["bindings"]
    identity = {}
    for name, binding_name in IDENTITY_BINDINGS.items():
        binding = bindings[binding_name]
        path = bound_path(binding)
        identity[name] = {"binding": binding_name, "path": _display_path(path),
                          "bytes": int(binding["bytes"]), "sha256": str(binding["sha256"])}
    return identity


def _save_array(directory: Path, name: str, values: np.ndarray, staging: Path) -> Mapping[str, object]:
    path = directory / (name + ".npy")
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return {"path": str(path.relative_to(staging)), "shape": list(values.shape),
            "dtype": str(values.dtype), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def materialize_c1_v2_fold(*, protocol: Mapping[str, object], fold_number: int,
                           fold_root: Path,
                           intervals_ms: Optional[Mapping[str, tuple]] = None,
                           smoke: bool = False) -> Mapping[str, object]:
    """Slice the frozen shared arrays into one sealed fold input directory.

    No schema, scaler, Drain3 or graph fitting is performed or permitted here.
    The writer lock is an exclusive directory creation, so an interrupted fold
    stays visibly incomplete and is never adopted or replaced.

    ``smoke=True`` (with explicit ``intervals_ms``) is the bounded smoke path; it
    refuses to write anywhere inside the formal run root and never uses the
    locked fold intervals silently.
    """
    g1_manifest = validate_g1_stage(protocol)
    fold_root = Path(fold_root).resolve()
    if fold_root.exists():
        raise FileExistsError("P6-C1-v2 fold output already exists")
    if smoke:
        if intervals_ms is None:
            raise ValueError("the smoke path requires explicit short intervals")
        if within(fold_root, ROOT / "experiments/p6/c1_supervision_oos"):
            raise ValueError("the smoke path cannot write inside the formal run root")
    else:
        if intervals_ms is not None:
            raise ValueError("formal folds must use the locked protocol intervals")
        if (fold_root.name != "fold_{:02d}".format(fold_number)
                or fold_root.parent.name != "folds"):
            raise ValueError("P6-C1-v2 fold output must be a new folds/fold_XX directory")
        if fold_root.parent != run_root(protocol) / "folds":
            raise ValueError("P6-C1-v2 fold output must live inside the frozen run root")

    bindings = protocol["bindings"]
    base = read_json(bound_path(bindings["base_config"]))
    if tuple(base["services"]) != tuple(GAIA_SERVICES):
        raise ValueError("shared preprocessing service order differs from the canonical order")
    grid_ms, window_bins = grid_geometry(protocol, base)
    if smoke:
        intervals = {}
        for name in SEGMENTS:
            start, end = (int(value) for value in intervals_ms[name])
            if (start >= end or start % grid_ms or end % grid_ms
                    or end - start <= window_bins * grid_ms):
                raise ValueError("smoke {} interval cannot host legal windows".format(name))
            intervals[name] = (start, end)
        if (intervals["fit"][1] != intervals["selection"][0]
                or intervals["selection"][1] != intervals["generation"][0]):
            raise ValueError("smoke segments must be adjacent and ordered")
    else:
        intervals = fold_intervals(protocol, fold_number, grid_ms=grid_ms,
                                   window_bins=window_bins)
    identity = _shared_identity(protocol)

    shared = {name: np.load(bound_path(bindings[IDENTITY_BINDINGS[name]]), mmap_mode="r",
                            allow_pickle=False)
              for name in ARRAY_BINDING_NAMES}
    metric, log, trace = shared["metric"], shared["log"], shared["trace"]
    train_timestamps = shared["train_timestamps"]
    graph = shared["graph"]
    if (metric.ndim != 3 or log.ndim != 3 or trace.ndim != 4
            or metric.shape[1] != len(GAIA_SERVICES) or log.shape[1] != len(GAIA_SERVICES)
            or trace.shape[1:3] != (len(GAIA_SERVICES), len(GAIA_SERVICES))
            or metric.shape[2] != EXPECTED_DIMENSIONS["raw_node"]
            or log.shape[2] != EXPECTED_DIMENSIONS["log_len"]
            or trace.shape[3] != EXPECTED_DIMENSIONS["raw_edge"]):
        raise ValueError("shared arrays do not satisfy the frozen 48/32/8 geometry")
    if graph.shape != (len(GAIA_SERVICES), len(GAIA_SERVICES)):
        raise ValueError("shared graph does not follow the canonical service order")
    if len(train_timestamps) != metric.shape[0] or len(train_timestamps) != log.shape[0] \
            or len(train_timestamps) != trace.shape[0]:
        raise ValueError("shared arrays are misaligned")
    if np.any(np.diff(np.asarray(train_timestamps)) != grid_ms):
        raise ValueError("shared Train timeline is not an exact 30 s grid")
    train_start = int(train_timestamps[0])
    train_end = int(train_timestamps[-1]) + grid_ms

    staging = fold_root
    staging.mkdir(parents=True, exist_ok=False)
    write_json_new(staging / "INCOMPLETE.json", {"status": "INCOMPLETE", "fold": fold_number})
    try:
        data_root = staging / "ad_data"
        data_root.mkdir()
        graph_record = _save_array(data_root, "graph", np.array(graph, copy=True), staging)
        segment_records = {}
        for name in SEGMENTS:
            start, end = intervals[name]
            if start < train_start or end > train_end:
                raise ValueError("fold {} {} leaves the frozen Train span".format(fold_number, name))
            first = start // grid_ms - train_start // grid_ms
            bins = (end - start) // grid_ms
            timestamps = np.asarray(train_timestamps[first:first + bins])
            if (len(timestamps) != bins
                    or not np.array_equal(timestamps, np.arange(start, end, grid_ms, dtype=np.int64))):
                raise ValueError("fold {} {} slice is not the locked grid".format(fold_number, name))
            legal = np.arange(bins - window_bins, dtype=np.int64)
            if len(legal) < 1:
                raise ValueError("fold {} {} cannot host a legal 300 s window".format(fold_number, name))
            metric_slice = np.array(metric[first:first + bins], dtype=np.float32)
            log_slice = np.array(log[first:first + bins], dtype=np.float32)
            trace_slice = np.array(trace[first:first + bins], dtype=np.float32)
            for label, values in (("metric", metric_slice), ("log", log_slice), ("trace", trace_slice)):
                if not np.isfinite(values).all():
                    raise ValueError("fold {} {} has non-finite {}".format(fold_number, name, label))
            directory = data_root / name
            directory.mkdir()
            files = {key: _save_array(directory, key, values, staging)
                     for key, values in (("timestamps", timestamps), ("metric", metric_slice),
                                         ("log", log_slice), ("trace", trace_slice),
                                         ("legal_window_indices", legal))}
            segment_records[name] = {
                "interval_ms": [start, end], "time_bins": int(bins),
                "window_bins": int(window_bins), "legal_windows": int(len(legal)),
                "first_prediction_available_ms": start + window_bins * grid_ms,
                "last_prediction_available_ms": end - grid_ms,
                "label_fields_present": [], "files": files,
            }
        if _shared_identity(protocol) != identity:
            raise ValueError("shared frozen inputs changed during fold slicing")
        if validate_g1_stage(protocol) != g1_manifest:
            raise ValueError("P6-C1-v2 G1 stage changed during fold slicing")
        manifest = {
            "schema_version": FOLD_MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "fold": int(fold_number),
            "fold_root": str(fold_root),
            "formal": not bool(smoke),
            "smoke": bool(smoke),
            "evidence_grade": "supervision_OOS_and_detector_parameter_OOS_only",
            "fit_semantics": "frozen_shared_original_train_label_free_preprocessing_sliced_by_locked_timestamps",
            "shared_preprocessing_refit": False,
            "schema_refit": False,
            "scaler_refit": False,
            "graph_refit": False,
            "drain3_refit": False,
            "decision_inputs": [],
            "node_anomaly_labels_read": False,
            "gt_labels_used_for_fold_inputs": False,
            "test_timestamps_used": False,
            "services": list(GAIA_SERVICES),
            "dimensions": dict(EXPECTED_DIMENSIONS),
            "shared_identity": identity,
            "graph": graph_record,
            "segments": segment_records,
            "g1_completion_manifest_sha256": sha256(
                run_root(protocol) / "g1_static/completion_manifest.json"),
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
        }
        write_json_new(staging / "completion_manifest.json", manifest)
        (staging / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as exc:
        try:
            write_json_new(staging / "failure.json",
                           {"status": "INCOMPLETE", "error_type": type(exc).__name__,
                            "message": str(exc)})
        except OSError:
            pass
        raise


def validate_c1_v2_fold(fold_root: Path, *, protocol: Mapping[str, object],
                        manifest: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    """Read-only seal check for one sliced fold; no shared array is re-hashed twice."""
    fold_root = Path(fold_root).resolve()
    if (fold_root / "INCOMPLETE.json").exists():
        raise ValueError("P6-C1-v2 fold output is incomplete")
    if manifest is None:
        manifest = read_json(fold_root / "completion_manifest.json")
    if (manifest.get("schema_version") != FOLD_MANIFEST_SCHEMA
            or manifest.get("status") != "COMPLETE"
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("run_id") != RUN_ID
            or manifest.get("fold_root") != str(fold_root)):
        raise ValueError("P6-C1-v2 fold completion identity drift")
    if tuple(manifest.get("services", ())) != tuple(GAIA_SERVICES):
        raise ValueError("P6-C1-v2 fold service order drift")
    if dict(manifest.get("dimensions", {})) != EXPECTED_DIMENSIONS:
        raise ValueError("P6-C1-v2 fold dimensions drift")
    if manifest.get("shared_preprocessing_refit") or manifest.get("schema_refit") \
            or manifest.get("scaler_refit") or manifest.get("graph_refit") \
            or manifest.get("drain3_refit"):
        raise ValueError("P6-C1-v2 fold input claims a refit")
    if manifest.get("shared_identity") != _shared_identity(protocol):
        raise ValueError("P6-C1-v2 fold shared-input identity drift")
    if manifest.get("g1_completion_manifest_sha256") != sha256(
            run_root(protocol) / "g1_static/completion_manifest.json"):
        raise ValueError("P6-C1-v2 fold G1 provenance drift")
    base = read_json(bound_path(protocol["bindings"]["base_config"]))
    grid_ms, window_bins = grid_geometry(protocol, base)
    if manifest.get("smoke"):
        if manifest.get("formal") is not False:
            raise ValueError("P6-C1-v2 smoke fold must not claim to be formal")
        intervals = {}
        for name in SEGMENTS:
            start, end = (int(value) for value in manifest["segments"][name]["interval_ms"])
            if (start >= end or start % grid_ms or end % grid_ms
                    or end - start <= window_bins * grid_ms):
                raise ValueError("P6-C1-v2 smoke segment is not window-legal: {}".format(name))
            intervals[name] = (start, end)
        if (intervals["fit"][1] != intervals["selection"][0]
                or intervals["selection"][1] != intervals["generation"][0]):
            raise ValueError("P6-C1-v2 smoke segments must be adjacent and ordered")
    else:
        intervals = fold_intervals(protocol, int(manifest["fold"]), grid_ms=grid_ms,
                                   window_bins=window_bins)
    for name in SEGMENTS:
        segment = manifest["segments"][name]
        start, end = intervals[name]
        if set(segment["files"]) != set(SEGMENT_FILES):
            raise ValueError("P6-C1-v2 fold segment files drift: {}".format(name))
        for entry in segment["files"].values():
            path = (fold_root / entry["path"]).resolve()
            if not within(path, fold_root) or not path.is_file():
                raise ValueError("P6-C1-v2 fold file missing: {}".format(entry["path"]))
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError("P6-C1-v2 fold output drift: {}".format(entry["path"]))
        timestamps = np.load(fold_root / segment["files"]["timestamps"]["path"],
                             mmap_mode="r", allow_pickle=False)
        legal = np.load(fold_root / segment["files"]["legal_window_indices"]["path"],
                        mmap_mode="r", allow_pickle=False)
        bins = (end - start) // grid_ms
        if (segment["interval_ms"] != [start, end] or segment["time_bins"] != bins
                or segment["window_bins"] != window_bins
                or segment["legal_windows"] != bins - window_bins
                or segment["first_prediction_available_ms"] != start + window_bins * grid_ms
                or segment["last_prediction_available_ms"] != end - grid_ms
                or not np.array_equal(timestamps, np.arange(start, end, grid_ms, dtype=np.int64))
                or not np.array_equal(legal, np.arange(bins - window_bins, dtype=np.int64))):
            raise ValueError("P6-C1-v2 fold segment geometry drift: {}".format(name))
    graph = np.load(fold_root / manifest["graph"]["path"], mmap_mode="r", allow_pickle=False)
    if graph.shape != (len(GAIA_SERVICES), len(GAIA_SERVICES)):
        raise ValueError("P6-C1-v2 fold graph geometry drift")
    return manifest


def load_segment(fold_root: Path, segment: str, *, manifest: Mapping[str, object]):
    """Memory-mapped segment arrays with the frozen service axis."""
    if segment not in SEGMENTS:
        raise ValueError("P6-C1-v2 segment must be fit, selection or generation")
    files = manifest["segments"][segment]["files"]
    arrays = {name: np.load(Path(fold_root) / files[name]["path"], mmap_mode="r",
                            allow_pickle=False) for name in ARRAY_NAMES}
    if (arrays["metric"].shape[1] != len(GAIA_SERVICES)
            or arrays["log"].shape[1] != len(GAIA_SERVICES)
            or arrays["trace"].shape[1:3] != (len(GAIA_SERVICES), len(GAIA_SERVICES))):
        raise ValueError("P6-C1-v2 fold arrays leave the canonical service axis")
    return arrays
