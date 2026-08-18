#!/usr/bin/env python3
"""Run nested five-fold OOF C0-M on frozen P2 whole-context metric features."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_manifest_cases
from src.evaluation import evaluate_ranking_report
from src.features import load_feature_matrices
from src.models import (
    IndependentLinearRanker,
    case_balanced_targets,
    rank_service_scores,
    select_feature_columns,
)


RUN_SCHEMA_VERSION = "p2_nested_oof_v1"
METHOD = "c0_m"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(case_ids) -> str:
    payload = "".join("{}\n".format(case_id) for case_id in sorted(case_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_line(record) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, records) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    rows = 0
    try:
        with temporary.open("wb") as handle:
            for record in records:
                rendered = _canonical_line(record)
                handle.write(rendered)
                digest.update(rendered)
                rows += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"rows": rows, "sha256": digest.hexdigest()}


def _write_json(path: Path, record) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bytes": len(rendered.encode("utf-8")), "sha256": _sha256(path)}


def _read_jsonl(path: Path) -> tuple:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _case_rows(index) -> dict:
    rows = {}
    for row_index, record in enumerate(index):
        rows.setdefault(record["case_id"], []).append(row_index)
    return {case_id: tuple(values) for case_id, values in rows.items()}


def _row_indices(case_ids, rows_by_case) -> np.ndarray:
    return np.fromiter(
        (
            row_index
            for case_id in sorted(case_ids)
            for row_index in rows_by_case[case_id]
        ),
        dtype=np.int64,
    )


def _subset_rows(index, row_indices) -> tuple:
    return tuple(index[int(row_index)] for row_index in row_indices)


def _root_macro_avg5(inputs, labels, rankings) -> float:
    return float(
        evaluate_ranking_report(inputs, labels, rankings)["root_service"][
            "macro"
        ]["Avg@5"]
    )


def _fit_and_rank(
    design,
    index,
    rows_by_case,
    labels_by_id,
    fit_ids,
    score_ids,
    regularization_c,
    random_seed,
):
    fit_rows = _row_indices(fit_ids, rows_by_case)
    score_rows = _row_indices(score_ids, rows_by_case)
    fit_index = _subset_rows(index, fit_rows)
    fit_labels = tuple(labels_by_id[case_id] for case_id in sorted(fit_ids))
    targets, weights = case_balanced_targets(fit_index, fit_labels)
    model = IndependentLinearRanker(regularization_c, random_seed).fit(
        design[fit_rows], targets, weights
    )
    scores = model.score(design[score_rows])
    score_index = _subset_rows(index, score_rows)
    rankings = rank_service_scores(score_index, scores)
    score_by_case = {}
    for record, score in zip(score_index, scores):
        score_by_case.setdefault(record["case_id"], {})[record["service"]] = float(
            score
        )
    return rankings, score_by_case, model.audit()


def _run_dataset(
    dataset_key: str,
    manifest_directory: Path,
    split_directory: Path,
    feature_directory: Path,
    output_directory: Path,
    inclusion_path: Path,
    c_values,
    random_seed: int,
):
    started = time.monotonic()
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included_ids = {
            row["case_id"] for row in _read_jsonl(inclusion_path)
        }
        inputs = tuple(row for row in inputs if row.case_id in included_ids)
        labels = tuple(row for row in labels if row.case_id in included_ids)
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}

    assignments = {
        row["case_id"]: row
        for row in _read_jsonl(split_directory / "assignments.jsonl")
        if row["case_id"] in inputs_by_id
    }
    if set(assignments) != set(inputs_by_id):
        raise ValueError("selected split does not cover model cohort")
    fold_by_case = {case_id: row["split"] for case_id, row in assignments.items()}
    group_by_case = {case_id: row["group_id"] for case_id, row in assignments.items()}
    folds = tuple(sorted(set(fold_by_case.values())))
    if len(folds) != 5:
        raise ValueError("C0-M requires the frozen five-fold assignment")

    feature_manifest, index, values, observed = load_feature_matrices(
        str(feature_directory)
    )
    if set(record["case_id"] for record in index) != set(inputs_by_id):
        raise ValueError("feature bundle does not match model cohort")
    feature_columns = select_feature_columns(
        feature_manifest["feature_names"], ("whole.",)
    )
    value_design = np.asarray(values[:, feature_columns], dtype=np.float32)
    mask_design = np.asarray(observed[:, feature_columns], dtype=np.float32)
    design = np.concatenate((value_design, mask_design), axis=1)
    if not np.isfinite(design).all():
        raise ValueError("C0-M design contains non-finite values")
    rows_by_case = _case_rows(index)

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
        outer_train_ids = tuple(
            sorted(set(inputs_by_id) - set(test_ids))
        )
        candidate_audit = []
        for regularization_c in c_values:
            inner_scores = []
            inner_audit = []
            for validation_fold in outer_train_folds:
                validation_ids = tuple(
                    sorted(
                        case_id
                        for case_id, fold in fold_by_case.items()
                        if fold == validation_fold
                    )
                )
                fit_ids = tuple(
                    sorted(set(outer_train_ids) - set(validation_ids))
                )
                fit_groups = {group_by_case[case_id] for case_id in fit_ids}
                validation_groups = {
                    group_by_case[case_id] for case_id in validation_ids
                }
                rankings, _, _ = _fit_and_rank(
                    design,
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
            candidate_audit.append(
                {
                    "C": float(regularization_c),
                    "inner_folds": inner_audit,
                    "mean_root_service_macro_Avg@5": float(
                        np.mean(inner_scores)
                    ),
                }
            )
        selected = sorted(
            candidate_audit,
            key=lambda row: (-row["mean_root_service_macro_Avg@5"], row["C"]),
        )[0]
        selected_c = selected["C"]
        outer_train_groups = {
            group_by_case[case_id] for case_id in outer_train_ids
        }
        test_groups = {group_by_case[case_id] for case_id in test_ids}
        rankings, scores, model_audit = _fit_and_rank(
            design,
            index,
            rows_by_case,
            labels_by_id,
            outer_train_ids,
            test_ids,
            selected_c,
            random_seed,
        )
        if set(oof_rankings) & set(rankings):
            raise ValueError("outer folds produced duplicate OOF cases")
        oof_rankings.update(rankings)
        oof_scores.update(scores)
        fold_audit.append(
            {
                "inner_candidates": candidate_audit,
                "model": model_audit,
                "outer_fold": outer_fold,
                "selected_C": selected_c,
                "test_case_count": len(test_ids),
                "test_case_ids_sha256": _id_digest(test_ids),
                "train_case_count": len(outer_train_ids),
                "train_case_ids_sha256": _id_digest(outer_train_ids),
                "train_test_group_overlap": len(outer_train_groups & test_groups),
                "train_test_overlap": len(set(outer_train_ids) & set(test_ids)),
            }
        )
        print(
            "[c0-m] {} outer fold {}/5 selected C={}".format(
                dataset_key, outer_index, selected_c
            ),
            flush=True,
        )
    if set(oof_rankings) != set(inputs_by_id):
        raise ValueError("C0-M did not produce complete OOF rankings")

    ordered_inputs = tuple(inputs_by_id[case_id] for case_id in sorted(inputs_by_id))
    ordered_labels = tuple(labels_by_id[case_id] for case_id in sorted(inputs_by_id))
    metrics = evaluate_ranking_report(ordered_inputs, ordered_labels, oof_rankings)
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
        "fit_scope": "outer-train labels only; inner validation labels only for C selection",
        "folds": fold_audit,
        "hyperparameter_objective": "mean inner root-service macro Avg@5",
        "label_firewall": {
            "feature_extraction_reads_labels": False,
            "prediction_records_contain_root_or_fault": False,
            "test_labels_used_for_fit_or_selection": False,
        },
        "tie_breaks": {
            "hyperparameter": "lower C (stronger regularization)",
            "ranking": "service name ascending",
        },
    }
    audit_file = _write_json(
        output_directory / "training_audit.json", training_audit
    )
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
            "design": "whole.* values concatenated with whole.* observed masks",
            "feature_column_count_before_masks": len(feature_columns),
            "random_seed": random_seed,
            "sample_weight": "0.5 root + 0.5 shared by non-roots per case",
            "scaler": "StandardScaler fit on current fit rows only",
            "solver": "sklearn LogisticRegression/liblinear",
        },
        "dataset": feature_manifest["dataset"],
        "files": {
            "metrics.json": metrics_file,
            "predictions.jsonl": predictions_file,
            "training_audit.json": audit_file,
        },
        "method": METHOD,
        "non_deterministic_files": {
            "runtime.json": {
                "role": "informational_runtime_excluded_from_core_checksums",
            }
        },
        "run_schema_version": RUN_SCHEMA_VERSION,
        "source": {
            "dataset_manifest_sha256": _sha256(
                manifest_directory / "manifest.json"
            ),
            "feature_manifest_sha256": _sha256(
                feature_directory / "manifest.json"
            ),
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
            raise ValueError("C0-M output checksum mismatch: {}".format(filename))
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
    parser.add_argument("--feature-root", default="artifacts/p2/features")
    parser.add_argument("--output-root", default="artifacts/p2/runs/c0_m")
    parser.add_argument("--summary", default="artifacts/p2/c0_m_summary.json")
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0)
    )
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    configs = {
        "gaia_main": {
            "feature": feature_root / "gaia_main" / "p2_metric_summary_v1",
            "inclusion": Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"),
            "manifest": manifest_root / "gaia",
            "split": split_root / "gaia",
        },
        "re2ob": {
            "feature": feature_root / "re2ob" / "p2_metric_summary_v1",
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
            config["feature"],
            output_root / dataset,
            config["inclusion"],
            tuple(sorted(set(args.c_values))),
            args.random_seed,
        )
        for dataset, config in configs.items()
    }
    _write_json(Path(args.summary), {"method": METHOD, "datasets": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
