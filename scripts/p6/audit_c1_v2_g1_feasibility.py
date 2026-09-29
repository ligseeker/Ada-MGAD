#!/usr/bin/env python3
"""P6-C1-v2 G1 static feasibility audit (read-only, no model training).

Verifies, before any detector exists, that the frozen shared Train arrays can
legally carry the three locked folds: coverage, monotonic 30 s grid, canonical
service order, dimensions, detector window legality, Test exclusion, and that
the frozen static GT/context upper bounds reproduce. Writes the sealed
``g1_static`` stage of the P6-C1-v2 run.

This script trains nothing, creates no OOS anchor, extracts no RCA feature and
never reads a Test label.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e2e.protocol import GAIA_SERVICES, SUPPORTED_FAULT_TYPES  # noqa: E402

CONFIG = ROOT / "configs/e2e/gaia_p6_c1_v2_supervision_oos.json"
CONFIG_SHA256 = "4e4529db1049fa5c04e658193b6b4e91eef0d9ad9e5d0efcd785dc776c475874"
PROTOCOL_ID = "P6-C1-SUPERVISION-OOS-v1"
RUN_ID = "c1-supervision-oos-v1-seed42"
SEGMENT_KEYS = {"fit": "detector_fit", "selection": "detector_selection",
                "generation": "anchor_generation"}
SEGMENTS = ("fit", "selection", "generation")
EXPECTED_DIMENSIONS = {"metric": 48, "log": 32, "trace": 8}
EXPECTED_TIME_BINS = {"metric": 60984, "log": 60984, "trace": 60984}
CONTEXT_MS = 300_000
TOLERANCE_MS = 60_000
GT_FEATURE_ROOT = ROOT / "data/p5/v3/rca_features_gt"
STAGE_FILES = ("input_manifest.json", "fold_manifest.json", "static_population.json",
               "integrity_report.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path: Path, payload) -> None:
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


def bound_path(binding) -> Path:
    path = (ROOT / str(binding["path"])).resolve()
    if not within(path, ROOT) or not path.is_file():
        raise ValueError("G1 bound input missing or outside repository: {}".format(path))
    if path.stat().st_size != int(binding["bytes"]) or sha256(path) != str(binding["sha256"]):
        raise ValueError("G1 bound input drift: {}".format(path))
    return path


def record(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def load_protocol() -> dict:
    if sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("P6-C1-v2 protocol config drift")
    protocol = read_json(CONFIG)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("run_id") != RUN_ID:
        raise ValueError("P6-C1-v2 protocol identity drift")
    return protocol


def fold_intervals(protocol: dict, fold_number: int, *, grid_ms: int, window_bins: int):
    if fold_number not in (1, 2, 3):
        raise ValueError("P6-C1-v2 has exactly three locked folds")
    fold = protocol["folds"][fold_number - 1]
    if int(fold["fold"]) != fold_number:
        raise ValueError("P6-C1-v2 fold identity drift")
    intervals = {}
    for name in SEGMENTS:
        start, end = (int(value) for value in fold["intervals_ms"][SEGMENT_KEYS[name]])
        intervals[name] = (start, end)
    for name in SEGMENTS:
        start, end = intervals[name]
        if (start >= end or start % grid_ms or end % grid_ms
                or end - start <= window_bins * grid_ms):
            raise ValueError("P6-C1-v2 {} interval is not window-legal".format(name))
    if (intervals["fit"][1] != intervals["selection"][0]
            or intervals["selection"][1] != intervals["generation"][0]):
        raise ValueError("P6-C1-v2 fold segments must be adjacent and ordered")
    return intervals


def read_legal_registry(pre_test_end_ms: int):
    """Label-legal, pre-Test GT rows; no Test row is retained or aggregated."""
    path = (ROOT / "artifacts/p5/v3/protocol/gt_event_registry.csv").resolve()
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["detector_domain"] == "True" and int(row["start_ms"]) < pre_test_end_ms:
                rows.append({"case_id": row["case_id"], "source_index": int(row["source_index"]),
                             "service": row["service"], "fault_type": row["fault_type"],
                             "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"])})
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("pre-Test GT registry identity drift")
    return rows


def counts_by(rows, field: str, universe) -> dict:
    found = {}
    for row in rows:
        found[row[field]] = found.get(row[field], 0) + 1
    if not set(found).issubset(set(universe)):
        raise ValueError("unexpected {} outside the frozen taxonomy".format(field))
    return {name: int(found.get(name, 0)) for name in universe}


def completed_in(rows, start_ms: int, end_ms: int):
    return [row for row in rows
            if row["start_ms"] >= start_ms and row["end_ms"] <= end_ms]


def array_facts(path: Path, name: str) -> dict:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    facts = {"dtype": str(values.dtype), "shape": list(values.shape),
             "bytes": path.stat().st_size, "sha256": sha256(path),
             "path": str(path.relative_to(ROOT))}
    if name in ("metric", "log"):
        if values.ndim != 3 or values.shape[1] != len(GAIA_SERVICES) or \
                int(values.shape[2]) != EXPECTED_DIMENSIONS[name]:
            raise ValueError("frozen {} array shape differs from the locked geometry".format(name))
    if name == "trace":
        if values.ndim != 4 or values.shape[1:3] != (len(GAIA_SERVICES), len(GAIA_SERVICES)) \
                or int(values.shape[3]) != EXPECTED_DIMENSIONS[name]:
            raise ValueError("frozen trace array shape differs from the locked geometry")
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    run_root = (ROOT / str(protocol["output_root"])).resolve()
    output = args.output_dir.resolve()
    if output != (run_root / "g1_static").resolve():
        raise ValueError("G1 output must be <run_root>/g1_static")
    if output.exists():
        raise FileExistsError("G1 output directory already exists")
    if within(output, ROOT / "experiments/p6/c1_detector_aligned"):
        raise ValueError("G1 must not write into the stopped prefix run")

    bindings = protocol["bindings"]
    bound = {name: bound_path(binding) for name, binding in bindings.items()}
    for name, path in bound.items():
        if path.name in ("labels.npy", "label_mask.npy"):
            raise ValueError("a node-anomaly label array must not be an input binding: {}".format(name))

    base = read_json(bound["base_config"])
    c0 = read_json(bound["c0_trigger_config"])
    if tuple(base["services"]) != tuple(GAIA_SERVICES):
        raise ValueError("shared preprocessing service order differs from the canonical order")
    if int(base["ad"]["grid_seconds"]) != 30 or int(base["ad"]["window_bins"]) != 10:
        raise ValueError("shared preprocessing grid/window geometry drift")
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    frozen_values = protocol["detector"]["expected_frozen_values"]
    checkable = [key for key in frozen_values if key != "pos_weight"]
    frozen_sources = {key: c0["training"].get(key) for key in checkable}
    frozen_sources["head_hidden"] = c0["model"].get("head_hidden")
    frozen_sources["graph_sparse_weight"] = base["ad_model"].get("graph_sparse_weight")
    for key in checkable:
        if frozen_sources.get(key) != frozen_values[key]:
            raise ValueError("C0 frozen training value drift: {}".format(key))
    if int(c0["seed"]) != int(protocol["seed"]):
        raise ValueError("C0/trigger seed differs from the locked protocol seed")
    if (int(c0["training"]["max_epochs"]) != int(protocol["resource_budget"]["max_epochs_per_fold"])
            or int(c0["training"]["num_workers"]) != int(protocol["resource_budget"]["detector_loader_workers"])):
        raise ValueError("C0 training budget differs from the locked protocol budget")

    schema = read_json(bound["shared_frozen_schema"])
    ad_manifest = read_json(bound["shared_ad_manifest"])
    if (schema.get("status") != "FROZEN" or schema.get("fit_split") != "train"
            or schema.get("decision_inputs") != ["train"]
            or schema.get("gt_labels_used") is not False
            or schema.get("test_used_for_selection") is not False):
        raise ValueError("shared preprocessing schema boundary drift")
    if (ad_manifest.get("status") != "COMPLETE"
            or ad_manifest.get("schema_sha256") != sha256(bound["shared_frozen_schema"])
            or tuple(ad_manifest.get("services", ())) != tuple(GAIA_SERVICES)):
        raise ValueError("shared AD manifest/schema binding drift")

    train_timestamps = np.load(bound["shared_train_timestamps"], mmap_mode="r", allow_pickle=False)
    test_timestamps = np.load(bound["shared_test_timestamps"], mmap_mode="r", allow_pickle=False)
    if train_timestamps.dtype != np.int64 or test_timestamps.dtype != np.int64:
        raise ValueError("frozen timestamps must be int64")
    if len(train_timestamps) != EXPECTED_TIME_BINS["metric"]:
        raise ValueError("frozen Train timeline length drift")
    if (np.diff(np.asarray(train_timestamps)) != grid_ms).any():
        raise ValueError("frozen Train timeline is not an exact 30 s grid")
    if (np.diff(np.asarray(test_timestamps)) != grid_ms).any():
        raise ValueError("frozen Test timeline is not an exact 30 s grid")
    train_start, train_end = int(train_timestamps[0]), int(train_timestamps[-1]) + grid_ms
    test_start = int(test_timestamps[0])
    if train_end != test_start or test_start != int(base["split"]["boundary_ms"]):
        raise ValueError("frozen Train/Test timeline boundary drift")

    metric = np.load(bound["shared_train_metric"], mmap_mode="r", allow_pickle=False)
    log = np.load(bound["shared_train_log"], mmap_mode="r", allow_pickle=False)
    trace = np.load(bound["shared_train_trace"], mmap_mode="r", allow_pickle=False)
    for name, values in (("metric", metric), ("log", log), ("trace", trace)):
        if values.shape[0] != len(train_timestamps) or values.dtype != np.float32:
            raise ValueError("frozen Train {} array is misaligned or not float32".format(name))
    if list(metric.shape) != [EXPECTED_TIME_BINS["metric"], len(GAIA_SERVICES), EXPECTED_DIMENSIONS["metric"]]:
        raise ValueError("frozen Train metric dimensions drift")
    if list(log.shape) != [EXPECTED_TIME_BINS["log"], len(GAIA_SERVICES), EXPECTED_DIMENSIONS["log"]]:
        raise ValueError("frozen Train log dimensions drift")
    if list(trace.shape) != [EXPECTED_TIME_BINS["trace"], len(GAIA_SERVICES), len(GAIA_SERVICES),
                             EXPECTED_DIMENSIONS["trace"]]:
        raise ValueError("frozen Train trace dimensions drift")

    graph = np.load(bound["shared_ad_graph"], allow_pickle=False)
    expected_graph = np.zeros((len(GAIA_SERVICES), len(GAIA_SERVICES)), dtype=np.float32)
    index_of = {service: index for index, service in enumerate(GAIA_SERVICES)}
    for source, destination in schema["traces"]["directed_edges"]:
        expected_graph[index_of[source], index_of[destination]] = 1.0
        expected_graph[index_of[destination], index_of[source]] = 1.0
    if graph.shape != expected_graph.shape or not np.array_equal(graph, expected_graph):
        raise ValueError("frozen detector graph differs from the shared schema edges")
    if (int(metric.shape[2]), int(log.shape[2]), int(trace.shape[3])) != (
            EXPECTED_DIMENSIONS["metric"], EXPECTED_DIMENSIONS["log"], EXPECTED_DIMENSIONS["trace"]):
        raise ValueError("frozen modality dimensions differ from 48/32/8")

    # ---- GT population (pre-Test GT registry; Test rows never read) --------
    registry_path = bound["gt_registry"]
    if sha256(registry_path) != base["event_registry"]["sha256"]:
        raise ValueError("GT registry binding differs from the base config")
    all_rows = read_legal_registry(test_start)
    gt_feature_ids = set(
        np.load(GT_FEATURE_ROOT / "case_ids.npy", allow_pickle=False).astype(str))
    gt_features_shape = list(np.load(GT_FEATURE_ROOT / "z2_features.npy",
                                     mmap_mode="r", allow_pickle=False).shape)
    if gt_features_shape[1:] != [len(GAIA_SERVICES), 68]:
        raise ValueError("P5 GT feature bundle geometry drift")

    # ---- per-fold static checks -------------------------------------------
    window_bins = int(base["ad"]["window_bins"])
    folds = []
    seen_cases = {}
    for fold_number in (1, 2, 3):
        intervals = fold_intervals(protocol, fold_number, grid_ms=grid_ms, window_bins=window_bins)
        locked = protocol["folds"][fold_number - 1]
        segments = {}
        for name in SEGMENTS:
            start, end = intervals[name]
            if start < train_start or end > train_end:
                raise ValueError("fold {} {} leaves the frozen Train span".format(fold_number, name))
            if start < test_start and end > test_start:
                raise ValueError("fold {} {} crosses the frozen Train/Test boundary".format(fold_number, name))
            time_bins = (end - start) // grid_ms
            timestamps = np.asarray(train_timestamps[start // grid_ms - train_start // grid_ms:
                                                     end // grid_ms - train_start // grid_ms])
            if len(timestamps) != time_bins or not np.array_equal(
                    timestamps, np.arange(start, end, grid_ms, dtype=np.int64)):
                raise ValueError("fold {} {} slice is not the locked grid".format(fold_number, name))
            legal_windows = time_bins - window_bins
            if legal_windows < 1:
                raise ValueError("fold {} {} cannot host a 300 s window".format(fold_number, name))
            first_index = start // grid_ms - train_start // grid_ms
            segment_metric = np.asarray(metric[first_index:first_index + time_bins])
            segment_log = np.asarray(log[first_index:first_index + time_bins])
            segment_trace = np.asarray(trace[first_index:first_index + time_bins])
            for label, values in (("metric", segment_metric), ("log", segment_log),
                                  ("trace", segment_trace)):
                if not np.isfinite(values).all():
                    raise ValueError("fold {} {} has non-finite {}".format(fold_number, name, label))
            segments[name] = {
                "interval_ms": [start, end],
                "iso_utc": [datetime.fromtimestamp(start / 1000, tz=timezone.utc).isoformat(),
                            datetime.fromtimestamp(end / 1000, tz=timezone.utc).isoformat()],
                "time_bins": int(time_bins),
                "legal_windows": int(legal_windows),
                "window_bins": int(window_bins),
                "first_prediction_available_ms": start + window_bins * grid_ms,
                "last_prediction_available_ms": end - grid_ms,
                "history_ms": window_bins * grid_ms,
                "grid_ms": grid_ms,
                "finite": True,
                "label_fields_present": [],
                "detector_input_fields": ["metric", "log", "trace"],
            }
        if (segments["generation"]["interval_ms"][0] < segments["selection"]["interval_ms"][1]
                or segments["selection"]["interval_ms"][0] < segments["fit"]["interval_ms"][1]):
            raise ValueError("fold {} segments are not strictly ordered".format(fold_number))

        fit_rows = completed_in(all_rows, *intervals["fit"])
        selection_rows = completed_in(all_rows, *intervals["selection"])
        generation_rows = completed_in(all_rows, *intervals["generation"])
        exclusions = {"no_existing_p5_gt_feature": 0, "gt_rca_context_crosses_generation_block": 0}
        eligible = []
        gen_start, gen_end = intervals["generation"]
        for row in generation_rows:
            if row["case_id"] not in gt_feature_ids:
                exclusions["no_existing_p5_gt_feature"] += 1
            elif row["start_ms"] - CONTEXT_MS < gen_start or row["start_ms"] + CONTEXT_MS > gen_end:
                exclusions["gt_rca_context_crosses_generation_block"] += 1
            else:
                eligible.append(row)
        if len(generation_rows) != len(eligible) + sum(exclusions.values()):
            raise ValueError("fold {} static GT accounting did not close".format(fold_number))
        for row in generation_rows:
            if row["case_id"] in seen_cases:
                raise ValueError("case {} appears in two generation folds".format(row["case_id"]))
            seen_cases[row["case_id"]] = fold_number
        safe_for_any_delay = sum(
            1 for row in eligible if row["start_ms"] + TOLERANCE_MS + CONTEXT_MS <= gen_end)
        reproduced = len(eligible)
        locked_bound = int(locked["static_gt_context_upper_bound"])
        if reproduced != locked_bound:
            raise ValueError(
                "fold {} static upper bound did not reproduce: {} != {}".format(
                    fold_number, reproduced, locked_bound))
        folds.append({
            "fold": fold_number,
            "segments": segments,
            "complete_gt": {"fit": len(fit_rows), "selection": len(selection_rows),
                            "generation": len(generation_rows)},
            "fit_complete_gt_fault_types": counts_by(fit_rows, "fault_type", SUPPORTED_FAULT_TYPES),
            "fit_complete_gt_services": counts_by(fit_rows, "service", GAIA_SERVICES),
            "generation_eligible_fault_types": counts_by(eligible, "fault_type", SUPPORTED_FAULT_TYPES),
            "generation_eligible_services": counts_by(eligible, "service", GAIA_SERVICES),
            "generation_exclusions": exclusions,
            "static_gt_context_upper_bound": reproduced,
            "generation_context_safe_for_any_0_to_60s_detection_delay": int(safe_for_any_delay),
            "minimum_actual_common_cases": int(locked["minimum_actual_common_cases"]),
        })

    generation_intervals = [tuple(fold["segments"]["generation"]["interval_ms"]) for fold in folds]
    for left in range(len(generation_intervals)):
        for right in range(left + 1, len(generation_intervals)):
            start_l, end_l = generation_intervals[left]
            start_r, end_r = generation_intervals[right]
            if min(end_l, end_r) > max(start_l, start_r):
                raise ValueError("generation intervals must not overlap")

    input_manifest = {
        "schema_version": "p6_c1_v2_g1_input_manifest_v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "config": record(CONFIG),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                            text=True).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "services": list(GAIA_SERVICES),
        "dimensions": {"metric": EXPECTED_DIMENSIONS["metric"], "log": EXPECTED_DIMENSIONS["log"],
                       "trace": EXPECTED_DIMENSIONS["trace"]},
        "train_timeline": {"first_ms": train_start, "last_bin_start_ms": train_end - grid_ms,
                           "end_ms": train_end, "time_bins": int(len(train_timestamps)),
                           "grid_ms": grid_ms},
        "test_timeline": {"first_ms": test_start, "last_bin_start_ms": int(test_timestamps[-1]),
                          "end_ms": int(test_timestamps[-1]) + grid_ms,
                          "time_bins": int(len(test_timestamps)), "grid_ms": grid_ms},
        "arrays": {
            "metric": array_facts(bound["shared_train_metric"], "metric"),
            "log": array_facts(bound["shared_train_log"], "log"),
            "trace": array_facts(bound["shared_train_trace"], "trace"),
            "train_timestamps": array_facts(bound["shared_train_timestamps"], "timestamps"),
            "test_timestamps": array_facts(bound["shared_test_timestamps"], "timestamps"),
            "graph": {"path": str(bound["shared_ad_graph"].relative_to(ROOT)),
                      "bytes": bound["shared_ad_graph"].stat().st_size,
                      "sha256": sha256(bound["shared_ad_graph"]),
                      "shape": list(graph.shape), "dtype": str(graph.dtype),
                      "matches_schema_edges": True},
        },
        "bound_inputs": {name: record(path) for name, path in bound.items()},
        "reference_inputs": {
            "p5_gt_feature_case_ids": record(GT_FEATURE_ROOT / "case_ids.npy"),
            "p5_gt_feature_bundle": {"path": str((GT_FEATURE_ROOT / "z2_features.npy").relative_to(ROOT)),
                                     "shape": gt_features_shape},
        },
        "detector_batch_fields": ["metric", "log", "trace"],
        "node_label_arrays_bound": False,
        "gt_or_root_fields_in_detector_input": False,
    }
    fold_manifest = {
        "schema_version": "p6_c1_v2_g1_fold_manifest_v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "seed": int(protocol["seed"]),
        "folds": folds,
        "generation_intervals_disjoint": True,
        "case_in_at_most_one_generation_fold": True,
    }
    static_population = {
        "schema_version": "p6_c1_v2_g1_static_population_v1",
        "protocol_id": PROTOCOL_ID,
        "pre_test_gt_registry_rows": len(all_rows),
        "static_upper_bound_sum": int(sum(fold["static_gt_context_upper_bound"] for fold in folds)),
        "floors_prespecified_before_execution": True,
        "per_fold": [{
            "fold": fold["fold"],
            "static_gt_context_upper_bound_reproduced": fold["static_gt_context_upper_bound"],
            "locked_static_gt_context_upper_bound": int(protocol["folds"][fold["fold"] - 1]["static_gt_context_upper_bound"]),
            "minimum_actual_common_cases": fold["minimum_actual_common_cases"],
            "generation_context_safe_for_any_0_to_60s_detection_delay": fold["generation_context_safe_for_any_0_to_60s_detection_delay"],
            "generation_exclusions": fold["generation_exclusions"],
        } for fold in folds],
        "interpretation": ("Static GT/context upper bounds and floors, not observed OOS anchors. "
                          "No detector score, episode or Test label was read."),
    }

    report_lines = [
        "# P6-C1-v2 G1 static feasibility report",
        "",
        "Status: **PASS**. Read-only static audit; no model was trained and no Test label was read.",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Frozen Train span | {} .. {} ({} bins) |".format(train_start, train_end, len(train_timestamps)),
        "| Frozen Test start | {} |".format(test_start),
        "| Grid | exact 30 s, monotonic, int64 |",
        "| Service order | canonical ten services, identical to shared manifest |",
        "| Dimensions | metric 48D, log 32D, trace 8D |",
        "| Graph | identical to the shared schema edges |",
        "| Detector batch fields | metric, log, trace only |",
        "| Node-anomaly label arrays | not bound, never read |",
        "| Window legality | every fold segment hosts >= 1 legal 300 s window |",
        "| Test isolation | no Test timestamp inside any fold segment |",
        "",
        "## Fold geometry and reproduced static bounds",
        "",
        "| Fold | Fit | Selection | Generation | Legal windows (F/S/G) | Static upper bound | Floor |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for fold in folds:
        segments = fold["segments"]
        report_lines.append("| {} | {} | {} | {} | {} / {} / {} | {} | {} |".format(
            fold["fold"],
            segments["fit"]["iso_utc"][0], segments["selection"]["iso_utc"][0],
            segments["generation"]["iso_utc"][0],
            segments["fit"]["legal_windows"], segments["selection"]["legal_windows"],
            segments["generation"]["legal_windows"],
            fold["static_gt_context_upper_bound"], fold["minimum_actual_common_cases"]))
    report_lines += [
        "",
        "Static upper bounds reproduce the frozen G1 ledger values "
        "(1,192 / 1,529 / 1,533). These are GT/context ceilings, not observed anchors.",
        "The safety margin for any 0-60 s detection delay is reported per fold in "
        "`static_population.json`; it is one lower than the ceiling for folds 2 and 3.",
        "",
        "No STOP condition triggered. The formal fold detector remains blocked until the "
        "G2/G3 gates (shared-array adapter, smoke and guards) pass.",
        "",
    ]

    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": "g1_static"})
    try:
        write_json_new(output / "input_manifest.json", input_manifest)
        write_json_new(output / "fold_manifest.json", fold_manifest)
        write_json_new(output / "static_population.json", static_population)
        with (output / "integrity_report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(report_lines))
        for name, binding in bindings.items():
            if sha256(bound[name]) != binding["sha256"]:
                raise ValueError("bound input changed during G1: {}".format(name))
        manifest = {
            "schema_version": "p6_c1_v2_g1_completion_v1",
            "stage": "g1_static",
            "status": "COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "run_id": RUN_ID,
            "config_sha256": sha256(CONFIG),
            "source": record(Path(__file__).resolve()),
            "inputs": {name: record(path) for name, path in bound.items()},
            "files": {name: record(output / name) for name in STAGE_FILES},
            "checks": {
                "train_arrays_cover_three_folds": True,
                "timestamps_strictly_monotonic_30s_grid": True,
                "service_order_identical": True,
                "dimensions_correct": True,
                "no_test_timestamp_in_fold_segments": True,
                "no_gt_fields_in_detector_batch": True,
                "detector_windows_legal": True,
                "rca_context_bounds_evaluable": True,
                "static_upper_bounds_reproduced": True,
            },
        }
        write_json_new(output / "completion_manifest.json", manifest)
        (output / "INCOMPLETE.json").unlink()
    except Exception as exc:
        try:
            write_json_new(output / "failure.json",
                           {"status": "INCOMPLETE", "error_type": type(exc).__name__,
                            "message": str(exc)})
        except OSError:
            pass
        raise
    print("PASS P6-C1-v2 G1 static feasibility:", output)


if __name__ == "__main__":
    main()
