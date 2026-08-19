#!/usr/bin/env python3
"""Run nested five-fold OOF M1-S with inner-selected 60/120 s onset."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_c0_metric import (
    RUN_SCHEMA_VERSION,
    _case_rows,
    _fit_and_rank,
    _id_digest,
    _read_jsonl,
    _root_macro_avg5,
    _sha256,
    _write_json,
    _write_jsonl,
)
from src.data import read_manifest_cases
from src.evaluation import evaluate_ranking_report
from src.features import load_feature_matrices
from src.models import select_feature_columns


METHOD = "m1_s"
SUPPORTED_ONSETS = (60, 120)


def _pair_index(index):
    return tuple((row["case_id"], row["service"]) for row in index)


def _load_stage_designs(feature_directories, onset_candidates):
    loaded = {}
    base_index = None
    dataset = None
    feature_sources = {}
    for modality, directory in feature_directories.items():
        manifest, index, values, observed = load_feature_matrices(str(directory))
        if base_index is None:
            base_index = index
            dataset = manifest["dataset"]
        elif _pair_index(index) != _pair_index(base_index):
            raise ValueError("M1-S feature row indices do not align")
        if manifest["dataset"] != dataset:
            raise ValueError("M1-S feature datasets do not align")
        loaded[modality] = (manifest, values, observed)
        feature_sources[modality] = {
            "feature_manifest_sha256": _sha256(directory / "manifest.json"),
            "path": str(directory),
        }

    whole_values = []
    whole_masks = []
    whole_counts = {}
    for modality in ("metric", "log", "trace"):
        manifest, values, observed = loaded[modality]
        columns = select_feature_columns(manifest["feature_names"], ("whole.",))
        whole_values.append(np.asarray(values[:, columns], dtype=np.float32))
        whole_masks.append(np.asarray(observed[:, columns], dtype=np.float32))
        whole_counts[modality] = len(columns)

    designs = {}
    stage_counts = {}
    for onset in onset_candidates:
        stage_values = []
        stage_masks = []
        stage_counts[str(onset)] = {}
        for modality in ("metric", "trace"):
            manifest, values, observed = loaded[modality]
            columns = select_feature_columns(
                manifest["feature_names"], ("stage{}.".format(onset),)
            )
            stage_values.append(np.asarray(values[:, columns], dtype=np.float32))
            stage_masks.append(np.asarray(observed[:, columns], dtype=np.float32))
            stage_counts[str(onset)][modality] = len(columns)
        design = np.concatenate(
            tuple(whole_values + stage_values + whole_masks + stage_masks), axis=1
        )
        if not np.isfinite(design).all():
            raise ValueError("M1-S design contains non-finite values")
        designs[onset] = design
    return (
        dataset,
        base_index,
        designs,
        whole_counts,
        stage_counts,
        feature_sources,
    )


def _run_dataset(
    dataset_key,
    manifest_directory,
    split_directory,
    feature_directories,
    output_directory,
    inclusion_path,
    c_values,
    onset_candidates,
    random_seed,
):
    started = time.monotonic()
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included = {row["case_id"] for row in _read_jsonl(inclusion_path)}
        inputs = tuple(row for row in inputs if row.case_id in included)
        labels = tuple(row for row in labels if row.case_id in included)
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}
    assignments = {
        row["case_id"]: row
        for row in _read_jsonl(split_directory / "assignments.jsonl")
        if row["case_id"] in inputs_by_id
    }
    if set(assignments) != set(inputs_by_id):
        raise ValueError("selected split does not cover M1-S cohort")
    fold_by_case = {case_id: row["split"] for case_id, row in assignments.items()}
    group_by_case = {case_id: row["group_id"] for case_id, row in assignments.items()}
    folds = tuple(sorted(set(fold_by_case.values())))
    if len(folds) != 5:
        raise ValueError("M1-S requires the frozen five-fold assignment")

    (
        dataset,
        index,
        designs,
        whole_counts,
        stage_counts,
        feature_sources,
    ) = _load_stage_designs(feature_directories, onset_candidates)
    if set(record["case_id"] for record in index) != set(inputs_by_id):
        raise ValueError("M1-S features do not match model cohort")
    rows_by_case = _case_rows(index)
    candidate_configs = tuple(
        (regularization_c, onset)
        for regularization_c in c_values
        for onset in onset_candidates
    )
    oof_rankings = {}
    oof_scores = {}
    fold_audit = []
    for outer_index, outer_fold in enumerate(folds, start=1):
        outer_train_folds = tuple(fold for fold in folds if fold != outer_fold)
        test_ids = tuple(
            sorted(
                case_id
                for case_id, fold in fold_by_case.items()
                if fold == outer_fold
            )
        )
        outer_train_ids = tuple(sorted(set(inputs_by_id) - set(test_ids)))
        candidate_audit = []
        for candidate_index, (regularization_c, onset) in enumerate(
            candidate_configs, start=1
        ):
            inner_scores = []
            inner_audit = []
            for inner_index, validation_fold in enumerate(
                outer_train_folds, start=1
            ):
                validation_ids = tuple(
                    sorted(
                        case_id
                        for case_id, fold in fold_by_case.items()
                        if fold == validation_fold
                    )
                )
                fit_ids = tuple(sorted(set(outer_train_ids) - set(validation_ids)))
                fit_groups = {group_by_case[case_id] for case_id in fit_ids}
                validation_groups = {
                    group_by_case[case_id] for case_id in validation_ids
                }
                rankings, _, _ = _fit_and_rank(
                    designs[onset],
                    index,
                    rows_by_case,
                    labels_by_id,
                    fit_ids,
                    validation_ids,
                    regularization_c,
                    random_seed,
                )
                validation_inputs = tuple(
                    inputs_by_id[case_id] for case_id in validation_ids
                )
                validation_labels = tuple(
                    labels_by_id[case_id] for case_id in validation_ids
                )
                score = _root_macro_avg5(
                    validation_inputs, validation_labels, rankings
                )
                inner_scores.append(score)
                inner_audit.append(
                    {
                        "fit_case_count": len(fit_ids),
                        "fit_case_ids_sha256": _id_digest(fit_ids),
                        "fit_validation_group_overlap": len(
                            fit_groups & validation_groups
                        ),
                        "fit_validation_overlap": len(
                            set(fit_ids) & set(validation_ids)
                        ),
                        "root_service_macro_Avg@5": score,
                        "validation_case_count": len(validation_ids),
                        "validation_case_ids_sha256": _id_digest(validation_ids),
                        "validation_fold": validation_fold,
                    }
                )
                print(
                    "[m1-s] {} outer {}/5 candidate {}/{} inner {}/4".format(
                        dataset_key,
                        outer_index,
                        candidate_index,
                        len(candidate_configs),
                        inner_index,
                    ),
                    flush=True,
                )
            candidate_audit.append(
                {
                    "C": float(regularization_c),
                    "inner_folds": inner_audit,
                    "mean_root_service_macro_Avg@5": float(np.mean(inner_scores)),
                    "onset_seconds": int(onset),
                }
            )
        selected = sorted(
            candidate_audit,
            key=lambda row: (
                -row["mean_root_service_macro_Avg@5"],
                row["C"],
                row["onset_seconds"],
            ),
        )[0]
        selected_c = selected["C"]
        selected_onset = selected["onset_seconds"]
        outer_train_groups = {
            group_by_case[case_id] for case_id in outer_train_ids
        }
        test_groups = {group_by_case[case_id] for case_id in test_ids}
        rankings, scores, model_audit = _fit_and_rank(
            designs[selected_onset],
            index,
            rows_by_case,
            labels_by_id,
            outer_train_ids,
            test_ids,
            selected_c,
            random_seed,
        )
        if set(oof_rankings) & set(rankings):
            raise ValueError("M1-S outer folds produced duplicate OOF cases")
        oof_rankings.update(rankings)
        oof_scores.update(scores)
        fold_audit.append(
            {
                "inner_candidates": candidate_audit,
                "model": model_audit,
                "outer_fold": outer_fold,
                "selected_C": selected_c,
                "selected_onset_seconds": selected_onset,
                "test_case_count": len(test_ids),
                "test_case_ids_sha256": _id_digest(test_ids),
                "train_case_count": len(outer_train_ids),
                "train_case_ids_sha256": _id_digest(outer_train_ids),
                "train_test_group_overlap": len(outer_train_groups & test_groups),
                "train_test_overlap": len(set(outer_train_ids) & set(test_ids)),
            }
        )
        print(
            "[m1-s] {} outer fold {}/5 selected C={} onset={}".format(
                dataset_key, outer_index, selected_c, selected_onset
            ),
            flush=True,
        )
    if set(oof_rankings) != set(inputs_by_id):
        raise ValueError("M1-S did not produce complete OOF rankings")

    ordered_ids = sorted(inputs_by_id)
    metrics = evaluate_ranking_report(
        tuple(inputs_by_id[case_id] for case_id in ordered_ids),
        tuple(labels_by_id[case_id] for case_id in ordered_ids),
        oof_rankings,
    )
    predictions_file = _write_jsonl(
        output_directory / "predictions.jsonl",
        (
            {
                "case_id": case_id,
                "method": METHOD,
                "ranking": list(oof_rankings[case_id]),
                "service_scores": dict(sorted(oof_scores[case_id].items())),
                "split": fold_by_case[case_id],
            }
            for case_id in sorted(oof_rankings)
        ),
    )
    metrics_file = _write_json(output_directory / "metrics.json", metrics)
    training_audit = {
        "fit_scope": "outer-train labels only; inner validation labels only for joint C/onset selection",
        "folds": fold_audit,
        "hyperparameter_objective": "mean inner root-service macro Avg@5",
        "label_firewall": {
            "feature_extraction_reads_labels": False,
            "prediction_records_contain_root_or_fault": False,
            "test_labels_used_for_fit_or_selection": False,
        },
        "tie_breaks": {
            "hyperparameter": "lower C, then shorter supported onset",
            "ranking": "service name ascending",
        },
    }
    audit_file = _write_json(output_directory / "training_audit.json", training_audit)
    _write_json(
        output_directory / "runtime.json",
        {
            "informational_only": True,
            "runtime_seconds": float(time.monotonic() - started),
        },
    )
    run_manifest = {
        "case_count": len(inputs_by_id),
        "config": {
            "C_candidates": list(c_values),
            "design": "C1-I whole M/L/T plus selected-onset metric+trace stage values and masks",
            "log_stage_excluded": "RE2 content-complete 0.794900 below frozen 0.80 threshold",
            "onset_candidates_seconds": list(onset_candidates),
            "random_seed": random_seed,
            "sample_weight": "0.5 root + 0.5 shared by non-roots per case",
            "scaler": "StandardScaler fit on current fit rows only",
            "solver": "sklearn LogisticRegression/liblinear",
            "stage_feature_columns_before_masks": stage_counts,
            "whole_feature_columns_before_masks": whole_counts,
        },
        "dataset": dataset,
        "files": {
            "metrics.json": metrics_file,
            "predictions.jsonl": predictions_file,
            "training_audit.json": audit_file,
        },
        "method": METHOD,
        "non_deterministic_files": {
            "runtime.json": {
                "role": "informational_runtime_excluded_from_core_checksums"
            }
        },
        "run_schema_version": RUN_SCHEMA_VERSION,
        "source": {
            "dataset_manifest_sha256": _sha256(manifest_directory / "manifest.json"),
            "features": feature_sources,
            "split_assignment_sha256": _sha256(
                split_directory / "assignments.jsonl"
            ),
            "split_manifest_sha256": _sha256(
                split_directory / "split_manifest.json"
            ),
        },
    }
    _write_json(output_directory / "run_manifest.json", run_manifest)
    for filename, expected in run_manifest["files"].items():
        if _sha256(output_directory / filename) != expected["sha256"]:
            raise ValueError("M1-S output checksum mismatch: {}".format(filename))
    return {
        "case_count": len(inputs_by_id),
        "fault_type_macro": metrics["fault_type"]["macro"],
        "overall": metrics["overall"],
        "root_service_macro": metrics["root_service"]["macro"],
        "run_manifest_sha256": _sha256(output_directory / "run_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--metric-root", default="artifacts/p2/features")
    parser.add_argument("--event-root", default="artifacts/p2/event_features")
    parser.add_argument("--output-root", default="artifacts/p2/runs/m1_s")
    parser.add_argument("--summary", default="artifacts/p2/m1_s_summary.json")
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0)
    )
    parser.add_argument(
        "--onset-candidates", type=int, nargs="+", default=SUPPORTED_ONSETS
    )
    args = parser.parse_args()
    c_values = tuple(sorted(set(args.c_values)))
    onsets = tuple(sorted(set(args.onset_candidates)))
    if onsets != SUPPORTED_ONSETS:
        raise ValueError("unified M1-S requires frozen onset candidates 60/120")
    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    metric_root = Path(args.metric_root)
    event_root = Path(args.event_root)
    output_root = Path(args.output_root)
    configs = {
        "gaia_main": {
            "features": {
                "metric": metric_root / "gaia_main" / "p2_metric_summary_v1",
                "log": event_root / "gaia_main" / "p2_log_l0_v1",
                "trace": event_root / "gaia_main" / "p2_trace_t0_v1",
            },
            "inclusion": Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"),
            "manifest": manifest_root / "gaia",
            "split": split_root / "gaia",
        },
        "re2ob": {
            "features": {
                "metric": metric_root / "re2ob" / "p2_metric_summary_v1",
                "log": event_root / "re2ob" / "p2_log_l0_v1",
                "trace": event_root / "re2ob" / "p2_trace_t0_v1",
            },
            "inclusion": None,
            "manifest": manifest_root / "re2ob",
            "split": split_root / "re2ob",
        },
    }
    summary = {
        dataset: _run_dataset(
            dataset,
            config["manifest"],
            config["split"],
            config["features"],
            output_root / dataset,
            config["inclusion"],
            c_values,
            onsets,
            args.random_seed,
        )
        for dataset, config in configs.items()
    }
    _write_json(
        Path(args.summary),
        {"datasets": summary, "method": METHOD, "run_schema_version": RUN_SCHEMA_VERSION},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
