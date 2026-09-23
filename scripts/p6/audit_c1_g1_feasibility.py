#!/usr/bin/env python3
"""Read-only, pre-Test C1 fold and GT-cohort feasibility inventory.

This script does not train a detector, create OOS anchors, build RCA features,
or evaluate predictions. Its fold geometry is exploratory, not a C1 protocol.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
TRIGGER = ROOT / "configs/e2e/gaia_p6_c0_system_trigger.json"
C0_SPLIT = ROOT / "experiments/p6/system_event_trigger/split_manifest.json"
C0_MANIFEST = ROOT / "experiments/p6/system_event_trigger/manifest.json"
C0F_COMPLETION = ROOT / (
    "experiments/p6/c0f_failure_audit/"
    "c0f-correction-20260923T0755Z/completion_manifest.json"
)
AD_SCHEMA = ROOT / "artifacts/p5/v3_preprocessing_v2/schema/frozen_preprocessing_schema.json"
AD_MANIFEST = ROOT / "artifacts/p5/v3_preprocessing_v2/ad/ad_data_manifest.json"
RCA_INDEX = ROOT / "data/p5/v3/rca_raw_index/index_manifest.json"
GT_REGISTRY = ROOT / "artifacts/p5/v3/rca/rca_case_registry_gt.csv"
GT_FEATURE_ROOT = ROOT / "data/p5/v3/rca_features_gt"
FAULT_TYPES = (
    "access_permission_denied", "cpu_anomalies", "file_moving",
    "login_failure", "memory_anomalies", "normal_memory_freed",
)
CONTEXT_MS = 300_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_pre_test_registry(path, pre_test_end_ms):
    selected = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["detector_domain"] == "True" and int(row["start_ms"]) < pre_test_end_ms:
                selected.append(row)
    return selected


def read_gt_case_identity(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({"case_id": row["case_id"], "split": row["split"],
                         "start_ms": row["start_ms"]})
    return rows


def counts(rows):
    found = Counter(row["fault_type"] for row in rows)
    if not set(found).issubset(FAULT_TYPES):
        raise ValueError("unknown fault type in Train rows")
    return {name: found[name] for name in FAULT_TYPES}


def service_counts(rows, services):
    found = Counter(row["service"] for row in rows)
    if not set(found).issubset(services):
        raise ValueError("unknown service in Train rows")
    return {name: found[name] for name in services}


def completed_in(rows, start_ms, end_ms):
    return [
        row for row in rows
        if int(row["start_ms"]) >= start_ms and int(row["end_ms"]) <= end_ms
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    try:
        output.relative_to(ROOT / "experiments/p6/c1_feasibility")
    except ValueError as exc:
        raise ValueError("output must be a new C1 feasibility run directory") from exc
    if output.exists():
        raise FileExistsError("G1 output directory already exists")

    config = read_json(CONFIG)
    trigger = read_json(TRIGGER)
    split = read_json(C0_SPLIT)
    c0 = read_json(C0_MANIFEST)
    c0f = read_json(C0F_COMPLETION)
    schema = read_json(AD_SCHEMA)
    ad = read_json(AD_MANIFEST)
    rca_index = read_json(RCA_INDEX)
    registry_path = ROOT / config["event_registry"]["path"]
    if sha256(registry_path) != config["event_registry"]["sha256"]:
        raise ValueError("frozen GT registry hash drift")
    if c0f["status"] != "COMPLETE" or c0f["c0_verdict"] != "BORDERLINE":
        raise ValueError("C0F completion/verdict drift")
    if sha256(C0F_COMPLETION) != "faabd0a5e6486a89c6e6e3cab1af8f146af6cdab9fe5e728c027f5d18f80745a":
        raise ValueError("C0F completion manifest SHA drift")
    if c0["checkpoint_sha256"] != "6a4317d6414774e1ca6be4e418ed3d47d678adfab1f0155e96e04aa3fa9981f5":
        raise ValueError("C0 checkpoint identity drift")
    if c0["source_artifacts"]["base_config"]["sha256"] != sha256(CONFIG):
        raise ValueError("C0/base-config binding drift")
    if c0["split_blocks"] != split["blocks"]:
        raise ValueError("C0 split-block manifest drift")
    if schema["status"] != "FROZEN" or schema["fit_split"] != "train" or schema["decision_inputs"] != ["train"]:
        raise ValueError("shared AD schema fit boundary drift")
    if ad["status"] != "COMPLETE" or ad["schema_sha256"] != sha256(AD_SCHEMA):
        raise ValueError("shared AD manifest/schema binding drift")

    start = int(config["split"]["absolute_start_ms"])
    end = int(config["split"]["absolute_end_ms"])
    grid_ms = int(config["ad"]["grid_seconds"]) * 1000
    if (end - start) % grid_ms:
        raise ValueError("timeline not aligned to frozen grid")
    bins = (end - start) // grid_ms
    boundaries = {index: start + (bins * index // 10) * grid_ms
                  for index in (0, 1, 2, 3, 4, 5, 6, 7, 10)}
    blocks = split["blocks"]
    if (boundaries[5] != int(blocks["fit"]["end_ms"])
            or boundaries[7] != int(blocks["test"]["start_ms"])
            or boundaries[7] != int(config["split"]["boundary_ms"])):
        raise ValueError("candidate geometry disagrees with frozen C0/P5 boundaries")
    if trigger["split"]["fit_fraction"] != [5, 10] or trigger["split"]["validation_cumulative_fraction"] != [7, 10]:
        raise ValueError("C0 split policy drift")

    # The unified registry is read only; no Test label row is retained or
    # aggregated. The existing GT bundle is used solely for Train ID coverage.
    train_rows = read_pre_test_registry(registry_path, boundaries[7])
    if len({row["case_id"] for row in train_rows}) != len(train_rows):
        raise ValueError("duplicate Train GT case identity")
    original_fit = completed_in(train_rows, boundaries[0], boundaries[5])
    original_validation = completed_in(train_rows, boundaries[5], boundaries[7])
    if (len(original_fit) != split["events"]["fit"]["ground_truth_events"]
            or len(original_validation) != split["events"]["validation"]["ground_truth_events"]):
        raise ValueError("Train GT population disagrees with frozen C0 split")
    complete_ids = {row["case_id"] for row in original_fit + original_validation}
    excluded = [row for row in train_rows if row["case_id"] not in complete_ids]
    if (len(excluded) != 1
            or [row["case_id"] for row in excluded] != split["purged_events"]["case_ids"]
            or not (int(excluded[0]["start_ms"]) < boundaries[7] < int(excluded[0]["end_ms"]))):
        raise ValueError("pre-Test GT exclusion is not the frozen Train/Test crossing")

    case_ids = np.load(GT_FEATURE_ROOT / "case_ids.npy", allow_pickle=False).astype(str)
    feature_splits = np.load(GT_FEATURE_ROOT / "splits.npy", allow_pickle=False).astype(str)
    anchors = np.load(GT_FEATURE_ROOT / "anchors_ms.npy", allow_pickle=False).astype(np.int64)
    features = np.load(GT_FEATURE_ROOT / "z2_features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(case_ids), 10, 68) or len(case_ids) != len(set(case_ids)):
        raise ValueError("P5 GT feature identity/shape drift")
    if not (len(feature_splits) == len(anchors) == len(case_ids)):
        raise ValueError("P5 GT feature metadata length drift")
    gt_cases = read_gt_case_identity(GT_REGISTRY)
    gt_by_id = {row["case_id"]: row for row in gt_cases}
    if len(gt_by_id) != len(gt_cases) or set(case_ids) != set(gt_by_id):
        raise ValueError("P5 GT registry/feature ID mismatch")
    for case_id, part, anchor in zip(case_ids, feature_splits, anchors):
        row = gt_by_id[case_id]
        if row["split"] != part or int(row["start_ms"]) != int(anchor):
            raise ValueError("P5 GT registry/feature metadata mismatch")
    feature_ids = set(case_ids)

    designs = {}
    for design_name, indices in (
        ("fit_only_3fold", (1, 2, 3)),
        ("all_train_4fold", (2, 3, 4, 5)),
    ):
        folds = []
        for number, fit_end_index in enumerate(indices, 1):
            fit = (boundaries[0], boundaries[fit_end_index])
            selection = (boundaries[fit_end_index], boundaries[fit_end_index + 1])
            generation = (boundaries[fit_end_index + 1], boundaries[fit_end_index + 2])
            fit_rows = completed_in(train_rows, *fit)
            selection_rows = completed_in(train_rows, *selection)
            generation_rows = completed_in(train_rows, *generation)
            exclusions = Counter()
            gt_eligible = []
            for row in generation_rows:
                onset = int(row["start_ms"])
                if row["case_id"] not in feature_ids:
                    exclusions["no_existing_p5_gt_feature"] += 1
                elif onset - CONTEXT_MS < generation[0] or onset + CONTEXT_MS > generation[1]:
                    exclusions["gt_rca_context_crosses_generation_block"] += 1
                else:
                    gt_eligible.append(row)
            if len(generation_rows) != len(gt_eligible) + sum(exclusions.values()):
                raise ValueError("static GT eligibility accounting did not close")
            folds.append({
                "fold": number,
                "intervals_ms": {"detector_fit": fit, "detector_selection": selection,
                                 "anchor_generation": generation},
                "complete_gt": {"fit": len(fit_rows), "selection": len(selection_rows),
                                "generation": len(generation_rows)},
                "fit_fault_types": counts(fit_rows),
                "fit_services": service_counts(fit_rows, config["services"]),
                "selection_fault_types": counts(selection_rows),
                "selection_services": service_counts(selection_rows, config["services"]),
                "generation_gt_feature_and_rca_context_eligible": len(gt_eligible),
                "generation_eligible_fault_types": counts(gt_eligible),
                "generation_eligible_services": service_counts(gt_eligible, config["services"]),
                "generation_context_safe_for_any_0_to_60s_detection_delay": sum(
                    int(row["start_ms"]) + 60_000 + CONTEXT_MS <= generation[1]
                    for row in gt_eligible
                ),
                "generation_exclusions": dict(sorted(exclusions.items())),
                "actual_oos_detected_anchors": None,
                "actual_common_train_cohort": None,
            })
        designs[design_name] = {
            "folds": folds,
            "gt_feature_and_context_eligible_sum": sum(
                fold["generation_gt_feature_and_rca_context_eligible"] for fold in folds
            ),
            "uses_original_c0_validation_as_generation": any(
                fold["intervals_ms"]["anchor_generation"][0] >= boundaries[5]
                for fold in folds
            ),
        }

    inputs = [CONFIG, TRIGGER, C0_SPLIT, C0_MANIFEST, C0F_COMPLETION,
              AD_SCHEMA, AD_MANIFEST, RCA_INDEX, registry_path, GT_REGISTRY]
    inputs += [GT_FEATURE_ROOT / name for name in
               ("case_ids.npy", "splits.npy", "anchors_ms.npy", "z2_features.npy")]
    result = {
        "schema_version": "p6_c1_g1_static_feasibility_v5",
        "status": "STATIC_INVENTORY_COMPLETE_EXECUTION_NO_GO_AS_IS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "audit_source": record(Path(__file__).resolve()),
        "input_files": {str(path.relative_to(ROOT)): record(path) for path in inputs},
        "timeline": {"absolute_start_ms": start, "absolute_end_ms": end,
                     "grid_ms": grid_ms, "total_bins": bins,
                     "pre_test_end_ms": boundaries[7],
                     "candidate_boundaries_ms": boundaries},
        "pre_test_population": {"detector_domain_registry_rows": len(train_rows),
                                "c0_fit_complete_gt": len(original_fit),
                                "c0_validation_complete_gt": len(original_validation),
                                "train_test_boundary_crossing": len(excluded),
                                "crossing_case_ids": [row["case_id"] for row in excluded]},
        "existing_gt_bundle": {"cases_all_splits": len(case_ids), "shape": list(features.shape),
                               "identity_and_anchor_alignment": "PASS"},
        "fold_design_status": "EXPLORATORY_FIXED_TIME_FRACTIONS_NOT_FROZEN_PROTOCOL",
        "preferred_static_candidate": "fit_only_3fold",
        "preferred_candidate_reason": "keeps fold selection and generation outside the original C0 Validation period; "
                                      "this avoids a known historical model-selection reuse, not a claim of new holdout data",
        "designs": designs,
        "data_boundary": {
            "ad_shared_schema_fit_interval_ms": [start, boundaries[7]],
            "ad_shared_schema_fit_source": "original 70 percent Train; future relative to internal C1 folds",
            "strict_forward_oos_from_current_ad_arrays": False,
            "rca_raw_index": "full raw timeline indexed; per-case values are exact-window local; "
                             "metric indicator set is selected from complete raw filename inventory",
        },
        "open_gates": [
            "prefix-fitted per-fold Metric/Log/Trace schema, graph and detector arrays are absent",
            "fold-specific detector checkpoints/selection and genuine OOS anchors are absent",
            "actual common legal GT/OOS-detected Train cohort size and class mix are unknown",
            "RCA raw-index metric-schema future-filename policy and fold boundary semantics need freezing",
            "strict fit/selection/generation window purge and fold-specific run/source locks need freezing",
        ],
        "interpretation": "GT feature/context counts are static upper bounds, not detected-anchor "
                          "counts or predicted C1 performance. Even fit-only folds remain historical development "
                          "data. No Test labels or model scores were analyzed.",
    }
    output.mkdir(parents=True, exist_ok=False)
    target = output / "feasibility_ledger.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(str(target))


if __name__ == "__main__":
    main()
