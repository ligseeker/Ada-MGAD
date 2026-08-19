#!/usr/bin/env python3
"""Independently audit the M1-S staged OOF run and compare it with C1-I."""

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_c0_metric import (
    RUN_SCHEMA_VERSION,
    _id_digest,
    _read_jsonl,
    _sha256,
    _write_json,
)
from src.data import read_manifest_cases
from src.evaluation import evaluate_ranking_report


AUDIT_SCHEMA_VERSION = "p2_m1_stage_audit_v1"
METHOD = "m1_s"
COMPARATOR = "c1_i"
METRIC_NAMES = ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
SECTIONS = ("fault_type_macro", "overall", "root_service_macro")
EXPECTED_ONSETS = (60, 120)
EXPECTED_C_VALUES = (0.01, 0.1, 1.0, 10.0)
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 4
EXPECTED_INNER_FITS = (
    EXPECTED_OUTER_FOLDS
    * len(EXPECTED_C_VALUES)
    * len(EXPECTED_ONSETS)
    * EXPECTED_INNER_FOLDS
)
EXPECTED_WHOLE_COLUMNS = {"log": 5, "metric": 17, "trace": 8}
EXPECTED_STAGE_COLUMNS = {"metric": 51, "trace": 24}
SECONDARY_GUARDRAIL = -0.01
LABEL_TOKENS = ("faulttype", "groundtruth", "rootservice")
PREDICTION_FIELDS = frozenset(
    {"case_id", "method", "ranking", "service_scores", "split"}
)


def compact_metrics(report):
    """Reduce a full ranking report to the three reported layers."""

    return {
        "fault_type_macro": report["fault_type"]["macro"],
        "overall": report["overall"],
        "root_service_macro": report["root_service"]["macro"],
    }


def metric_delta(actual, reference):
    """Signed actual-minus-reference delta at every reported layer."""

    return {
        section: {
            metric: float(actual[section][metric] - reference[section][metric])
            for metric in METRIC_NAMES
        }
        for section in SECTIONS
    }


def reconstruct_selection(candidates):
    """Re-derive the frozen inner choice: best score, lower C, shorter onset."""

    if not candidates:
        raise ValueError("M1-S outer fold records no inner candidates")
    grid = tuple(sorted((row["C"], row["onset_seconds"]) for row in candidates))
    expected_grid = tuple(
        sorted(
            (value, onset)
            for value in EXPECTED_C_VALUES
            for onset in EXPECTED_ONSETS
        )
    )
    if grid != expected_grid:
        raise ValueError(
            "M1-S inner candidates are not the frozen four C by two onset grid"
        )
    return sorted(
        candidates,
        key=lambda row: (
            -row["mean_root_service_macro_Avg@5"],
            row["C"],
            row["onset_seconds"],
        ),
    )[0]


def verify_design_width(config):
    """Check the recorded M1-S design against the frozen feature widths."""

    if tuple(config["onset_candidates_seconds"]) != EXPECTED_ONSETS:
        raise ValueError("M1-S onset candidates are not the frozen 60/120 s pair")
    if tuple(config["C_candidates"]) != EXPECTED_C_VALUES:
        raise ValueError("M1-S C candidates are not the frozen grid")
    if not config.get("log_stage_excluded"):
        raise ValueError("M1-S must record why log staged channels are excluded")
    if config["whole_feature_columns_before_masks"] != EXPECTED_WHOLE_COLUMNS:
        raise ValueError("M1-S whole feature widths are not the frozen 17/5/8")
    stage = config["stage_feature_columns_before_masks"]
    expected_keys = tuple(sorted(str(onset) for onset in EXPECTED_ONSETS))
    if tuple(sorted(stage)) != expected_keys:
        raise ValueError("M1-S stage widths do not cover exactly the frozen onsets")
    for counts in stage.values():
        if counts != EXPECTED_STAGE_COLUMNS:
            raise ValueError("M1-S stage feature widths are not the frozen 51/24")
    whole_values = sum(EXPECTED_WHOLE_COLUMNS.values())
    stage_values = sum(EXPECTED_STAGE_COLUMNS.values())
    return {
        "design_column_count": 2 * (whole_values + stage_values),
        "stage_value_columns": stage_values,
        "value_columns": whole_values + stage_values,
        "whole_value_columns": whole_values,
    }


def verify_fold_audit(folds, fold_case_ids):
    """Re-derive fold membership, isolation and selection from the audit trail."""

    if len(folds) != EXPECTED_OUTER_FOLDS:
        raise ValueError("M1-S training audit must contain five outer folds")
    fold_names = tuple(sorted(fold_case_ids))
    cohort = tuple(
        sorted(case_id for ids in fold_case_ids.values() for case_id in ids)
    )
    if len(set(cohort)) != len(cohort):
        raise ValueError("frozen folds assign at least one case more than once")

    seen = []
    inner_fits = 0
    selected_c = []
    selected_onsets = []
    inner_overlaps = []
    inner_group_overlaps = []
    for fold in folds:
        name = fold["outer_fold"]
        if name not in fold_case_ids:
            raise ValueError("M1-S outer fold is absent from the frozen split")
        if name in seen:
            raise ValueError("M1-S repeats an outer fold")
        seen.append(name)
        test_ids = fold_case_ids[name]
        test_set = set(test_ids)
        train_ids = tuple(case_id for case_id in cohort if case_id not in test_set)
        if fold["test_case_ids_sha256"] != _id_digest(test_ids):
            raise ValueError("recorded outer test IDs do not match the frozen split")
        if fold["train_case_ids_sha256"] != _id_digest(train_ids):
            raise ValueError("recorded outer train IDs do not match the frozen split")
        if fold["test_case_count"] != len(test_ids):
            raise ValueError("recorded outer test case count mismatch")
        if fold["train_case_count"] != len(train_ids):
            raise ValueError("recorded outer train case count mismatch")
        if fold["train_test_overlap"] or fold["train_test_group_overlap"]:
            raise ValueError("M1-S outer fold leakage detected")

        selected = reconstruct_selection(fold["inner_candidates"])
        if fold["selected_C"] != selected["C"]:
            raise ValueError("recorded selected C does not match the inner objective")
        if fold["selected_onset_seconds"] != selected["onset_seconds"]:
            raise ValueError(
                "recorded selected onset does not match the inner objective"
            )
        if fold["selected_onset_seconds"] not in EXPECTED_ONSETS:
            raise ValueError("M1-S selected an onset outside the frozen candidates")
        selected_c.append(fold["selected_C"])
        selected_onsets.append(fold["selected_onset_seconds"])

        expected_validation = tuple(other for other in fold_names if other != name)
        for candidate in fold["inner_candidates"]:
            inner_folds = candidate["inner_folds"]
            if len(inner_folds) != EXPECTED_INNER_FOLDS:
                raise ValueError("candidate does not cover four inner folds")
            validation = tuple(sorted(row["validation_fold"] for row in inner_folds))
            if validation != expected_validation:
                raise ValueError(
                    "inner rotation does not use exactly the four outer-train folds"
                )
            for inner in inner_folds:
                inner_fits += 1
                inner_overlaps.append(inner["fit_validation_overlap"])
                inner_group_overlaps.append(inner["fit_validation_group_overlap"])
                if inner["fit_validation_overlap"] or inner[
                    "fit_validation_group_overlap"
                ]:
                    raise ValueError("M1-S inner fold leakage detected")
                validation_ids = fold_case_ids[inner["validation_fold"]]
                if inner["validation_case_ids_sha256"] != _id_digest(validation_ids):
                    raise ValueError(
                        "inner validation IDs do not match the frozen split"
                    )
                validation_set = set(validation_ids)
                fit_ids = tuple(
                    case_id
                    for case_id in train_ids
                    if case_id not in validation_set
                )
                if inner["fit_case_ids_sha256"] != _id_digest(fit_ids):
                    raise ValueError("inner fit IDs do not match the frozen split")
                if inner["fit_case_count"] != len(fit_ids):
                    raise ValueError("recorded inner fit case count mismatch")

    if tuple(sorted(seen)) != fold_names:
        raise ValueError("M1-S outer folds do not cover the frozen split exactly once")
    if inner_fits != EXPECTED_INNER_FITS:
        raise ValueError(
            "M1-S must record exactly {} inner fits".format(EXPECTED_INNER_FITS)
        )
    return {
        "inner_fit_count": inner_fits,
        "inner_fit_validation_group_overlap_max": max(inner_group_overlaps),
        "inner_fit_validation_overlap_max": max(inner_overlaps),
        "outer_fold_count": len(folds),
        "outer_fold_ids_match_frozen_split": True,
        "selected_C_by_outer_fold": selected_c,
        "selected_onset_seconds_by_outer_fold": selected_onsets,
        "test_group_overlap_max": max(
            fold["train_test_group_overlap"] for fold in folds
        ),
        "test_overlap_max": max(fold["train_test_overlap"] for fold in folds),
    }


def _load_cohort(manifest_directory, inclusion_path):
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included = {row["case_id"] for row in _read_jsonl(inclusion_path)}
        inputs = tuple(row for row in inputs if row.case_id in included)
        labels = tuple(row for row in labels if row.case_id in included)
    return (
        {row.case_id: row for row in inputs},
        {row.case_id: row for row in labels},
    )


def _fold_case_ids(split_directory, cohort_ids):
    assignments = {
        row["case_id"]: row
        for row in _read_jsonl(split_directory / "assignments.jsonl")
        if row["case_id"] in cohort_ids
    }
    if set(assignments) != set(cohort_ids):
        raise ValueError("frozen split does not cover the audited cohort")
    grouped = {}
    for case_id, row in assignments.items():
        grouped.setdefault(row["split"], []).append(case_id)
    fold_case_ids = {
        fold: tuple(sorted(case_ids)) for fold, case_ids in grouped.items()
    }
    if len(fold_case_ids) != EXPECTED_OUTER_FOLDS:
        raise ValueError("frozen split does not contain five folds")
    fold_by_case = {
        case_id: row["split"] for case_id, row in assignments.items()
    }
    return fold_case_ids, fold_by_case


def _verify_run_files(run_directory, method):
    manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("run_schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run schema for {}".format(method))
    if manifest.get("method") != method:
        raise ValueError("run manifest method mismatch for {}".format(method))
    for filename, expected in manifest["files"].items():
        if _sha256(run_directory / filename) != expected["sha256"]:
            raise ValueError("run file checksum mismatch: {}".format(filename))
    return manifest


def _load_rankings(run_directory, method, cohort_ids, fold_by_case):
    prediction_path = run_directory / "predictions.jsonl"
    normalized = re.sub(
        r"[^a-z0-9]", "", prediction_path.read_text(encoding="utf-8").lower()
    )
    if any(token in normalized for token in LABEL_TOKENS):
        raise ValueError("prediction records contain label tokens")
    predictions = _read_jsonl(prediction_path)
    if any(set(record) != PREDICTION_FIELDS for record in predictions):
        raise ValueError("prediction record schema mismatch")
    if any(record["method"] != method for record in predictions):
        raise ValueError("prediction method mismatch")
    if any(
        record["split"] != fold_by_case[record["case_id"]] for record in predictions
    ):
        raise ValueError("prediction fold labels do not match the frozen split")
    rankings = {row["case_id"]: tuple(row["ranking"]) for row in predictions}
    if len(rankings) != len(predictions):
        raise ValueError("OOF predictions contain duplicate case IDs")
    if set(rankings) != set(cohort_ids):
        raise ValueError("OOF predictions do not cover the cohort exactly")
    return rankings


def _recompute(run_directory, inputs_by_id, labels_by_id, rankings):
    ordered_ids = sorted(inputs_by_id)
    recomputed = evaluate_ranking_report(
        tuple(inputs_by_id[case_id] for case_id in ordered_ids),
        tuple(labels_by_id[case_id] for case_id in ordered_ids),
        rankings,
    )
    recorded = json.loads(
        (run_directory / "metrics.json").read_text(encoding="utf-8")
    )
    if recomputed != recorded:
        raise ValueError(
            "recorded metrics for {} do not exactly recompute".format(run_directory)
        )
    return recorded


def _audit_dataset(dataset, manifest_directory, split_directory, run_root, inclusion):
    inputs_by_id, labels_by_id = _load_cohort(manifest_directory, inclusion)
    fold_case_ids, fold_by_case = _fold_case_ids(split_directory, set(inputs_by_id))

    run_directory = run_root / METHOD / dataset
    run_manifest = _verify_run_files(run_directory, METHOD)
    design = verify_design_width(run_manifest["config"])
    if run_manifest["config"]["random_seed"] != 20260819:
        raise ValueError("M1-S did not use the frozen seed")
    if run_manifest["case_count"] != len(inputs_by_id):
        raise ValueError("M1-S run manifest case count does not match the cohort")
    rankings = _load_rankings(run_directory, METHOD, set(inputs_by_id), fold_by_case)
    recorded = _recompute(run_directory, inputs_by_id, labels_by_id, rankings)

    training = json.loads(
        (run_directory / "training_audit.json").read_text(encoding="utf-8")
    )
    firewall = training["label_firewall"]
    if any(firewall.values()):
        raise ValueError("M1-S training audit reports a Label Firewall violation")
    fold_report = verify_fold_audit(training["folds"], fold_case_ids)

    comparator_directory = run_root / COMPARATOR / dataset
    _verify_run_files(comparator_directory, COMPARATOR)
    comparator_rankings = _load_rankings(
        comparator_directory, COMPARATOR, set(inputs_by_id), fold_by_case
    )
    comparator_recorded = _recompute(
        comparator_directory, inputs_by_id, labels_by_id, comparator_rankings
    )

    metrics = compact_metrics(recorded)
    comparator_metrics = compact_metrics(comparator_recorded)
    return {
        "audit": {
            "case_count": len(inputs_by_id),
            "comparator_core_files_verified": True,
            "comparator_metrics_exactly_recomputed": True,
            "core_files_verified": True,
            "label_firewall_flags_all_false": True,
            "label_free_predictions": True,
            "metrics_exactly_recomputed": True,
            "prediction_folds_match_frozen_split": True,
            **fold_report,
        },
        "comparator_metrics": comparator_metrics,
        "design": design,
        "metrics": metrics,
    }


def compare_with_comparator(metrics, comparator_metrics):
    """Build the H1 delta block the paired bootstrap cross-checks against."""

    delta = metric_delta(metrics, comparator_metrics)
    root = delta["root_service_macro"]
    return {
        "m1_s_delta": delta,
        "m1_s_primary_improved": bool(root["Avg@5"] > 0.0),
        "m1_s_secondary_within_guardrail": bool(root["AC@1"] >= SECONDARY_GUARDRAIL),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--run-root", default="artifacts/p2/runs")
    parser.add_argument("--summary", default="artifacts/p2/m1_s_summary.json")
    parser.add_argument("--output", default="artifacts/p2/m1_s_audit.json")
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    run_root = Path(args.run_root)
    configs = {
        "gaia_main": {
            "inclusion": Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"),
            "manifest": manifest_root / "gaia",
            "split": split_root / "gaia",
        },
        "re2ob": {
            "inclusion": None,
            "manifest": manifest_root / "re2ob",
            "split": split_root / "re2ob",
        },
    }
    missing = tuple(
        str(path)
        for path in (
            Path(args.summary),
            run_root / METHOD,
            run_root / COMPARATOR,
        )
        if not path.exists()
    )
    if missing:
        raise FileNotFoundError(
            "required source bindings are missing: {}".format(", ".join(missing))
        )
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary.get("method") != METHOD:
        raise ValueError("M1-S summary method mismatch")
    if summary.get("run_schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported M1-S summary schema")

    datasets = {}
    comparison = {}
    for dataset, config in configs.items():
        result = _audit_dataset(
            dataset,
            config["manifest"],
            config["split"],
            run_root,
            config["inclusion"],
        )
        recorded_summary = summary["datasets"][dataset]
        for section in SECTIONS:
            if recorded_summary[section] != result["metrics"][section]:
                raise ValueError(
                    "M1-S summary does not match the recomputed {} metrics".format(
                        dataset
                    )
                )
        if recorded_summary["case_count"] != result["audit"]["case_count"]:
            raise ValueError("M1-S summary case count mismatch")
        expected_manifest_sha256 = _sha256(
            run_root / METHOD / dataset / "run_manifest.json"
        )
        if recorded_summary["run_manifest_sha256"] != expected_manifest_sha256:
            raise ValueError("M1-S summary run manifest digest mismatch")
        comparison[dataset] = compare_with_comparator(
            result["metrics"], result["comparator_metrics"]
        )
        datasets[dataset] = result

    output = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "c1_i_comparison": comparison,
        "comparator": COMPARATOR,
        "datasets": datasets,
        "h1_exploratory_signal_both_datasets": all(
            comparison[dataset]["m1_s_primary_improved"] for dataset in configs
        ),
        "method": METHOD,
        "primary_endpoint": "root_service_macro Avg@5",
        "secondary_endpoint": "root_service_macro AC@1",
    }
    _write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
