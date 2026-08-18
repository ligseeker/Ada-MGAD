#!/usr/bin/env python
"""Freeze the GAIA service-single-root cohort and cohort-specific baselines."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p1_metric_change import _coverage_audit, _coverage_slice_reports
from scripts.run_p1_sanity_baselines import (
    BASELINE_SCHEMA_VERSION,
    _id_digest,
    _load_split,
    _read_jsonl,
    _write_baseline_result,
    _write_json,
    _write_jsonl,
)
from src.baselines import (
    deterministic_random_ranking,
    fit_root_frequency,
    metric_change_prediction,
)
from src.data import (
    CaseGroup,
    LabeledEventInterval,
    build_event_purity_flags,
)
from src.evaluation import evaluate_ranking_report, predict_rankings


INCLUSION_SCHEMA_VERSION = "p1_gaia_inclusion_v1"
MAIN_COHORT = "anchor_unique_root_service"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(case_ids, labels_by_id):
    return {
        "fault_type": dict(
            sorted(Counter(labels_by_id[case_id].fault_type for case_id in case_ids).items())
        ),
        "root_service": dict(
            sorted(
                Counter(
                    labels_by_id[case_id].root_service for case_id in case_ids
                ).items()
            )
        ),
    }


def _compact(metrics, manifest_sha):
    return {
        "fault_type_macro": metrics["fault_type"]["macro"],
        "overall": metrics["overall"],
        "root_service_macro": metrics["root_service"]["macro"],
        "run_manifest_sha256": manifest_sha,
    }


def _load_inventory_metric_predictions(path):
    rankings = {}
    details = {}
    for row in _read_jsonl(path):
        case_id = str(row["case_id"])
        rankings[case_id] = tuple(row["ranking"])
        details[case_id] = {
            "all_services_fallback": bool(row["all_services_fallback"]),
            "fallback_services": list(row["fallback_services"]),
            "observed_feature_counts": dict(row["observed_feature_counts"]),
            "service_scores": dict(row["service_scores"]),
        }
    return rankings, details


def finalize(args):
    manifest_directory = Path(args.manifest_root) / "gaia"
    split_directory = Path(args.split_root) / "gaia"
    (
        dataset_manifest,
        split_manifest,
        inputs,
        labels,
        assignments,
    ) = _load_split(manifest_directory, split_directory)
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}
    assignment_by_id = {row.case_id: row.split for row in assignments}

    audit_rows = _read_jsonl(manifest_directory / "event_audit.jsonl")
    group_rows = _read_jsonl(manifest_directory / "groups.jsonl")
    audit_by_id = {str(row["case_id"]): row for row in audit_rows}
    groups = tuple(
        CaseGroup(
            str(row["case_id"]),
            str(row["group_id"]),
            int(row["group_size"]),
            int(row["overlap_degree"]),
        )
        for row in group_rows
    )
    group_by_id = {row.case_id: row for row in groups}
    events = tuple(
        LabeledEventInterval(
            case_id=case_id,
            start_ms=int(audit_by_id[case_id]["start_ms"]),
            end_ms=int(audit_by_id[case_id]["end_ms"]),
            root_service=labels_by_id[case_id].root_service,
            fault_type=str(labels_by_id[case_id].fault_type),
        )
        for case_id in sorted(inputs_by_id)
    )
    flags = build_event_purity_flags(events, groups, context_radius_ms=300000)
    flags_by_id = {row.case_id: row for row in flags}
    for case_id, flag in flags_by_id.items():
        if flag.actual_overlap_degree != len(
            audit_by_id[case_id]["overlapping_case_ids"]
        ):
            raise ValueError("computed actual overlap disagrees with event audit")

    main_ids = sorted(
        case_id
        for case_id, flag in flags_by_id.items()
        if not flag.anchor_has_multiple_root_services
    )
    sensitivity_ids = sorted(set(inputs_by_id) - set(main_ids))
    same_root_concurrent_ids = sorted(
        case_id
        for case_id, flag in flags_by_id.items()
        if flag.anchor_concurrent_degree > 0
        and not flag.anchor_has_multiple_root_services
    )
    anchor_no_concurrent_ids = sorted(
        case_id
        for case_id, flag in flags_by_id.items()
        if flag.anchor_concurrent_degree == 0
    )
    actual_isolated_ids = sorted(
        case_id
        for case_id, flag in flags_by_id.items()
        if flag.actual_overlap_degree == 0
    )

    inclusion_directory = Path(args.inclusion_output_root)
    flag_records = tuple(
        {
            "actual_overlap_degree": flag.actual_overlap_degree,
            "anchor_concurrent_degree": flag.anchor_concurrent_degree,
            "anchor_concurrent_other_root_count": flag.anchor_concurrent_other_root_count,
            "anchor_has_multiple_root_services": flag.anchor_has_multiple_root_services,
            "case_id": flag.case_id,
            "context_group_fault_type_count": flag.context_group_fault_type_count,
            "context_group_root_service_count": flag.context_group_root_service_count,
            "context_group_size": flag.context_group_size,
            "context_overlap_degree": flag.context_overlap_degree,
            "main_cohort": not flag.anchor_has_multiple_root_services,
            "other_event_starts_in_context": flag.other_event_starts_in_context,
        }
        for flag in flags
    )
    cohort_records = tuple(
        {
            "case_id": case_id,
            "cohort": MAIN_COHORT,
            "group_id": group_by_id[case_id].group_id,
            "split": assignment_by_id[case_id],
        }
        for case_id in main_ids
    )
    flag_file = _write_jsonl(inclusion_directory / "flags.jsonl", flag_records)
    cohort_file = _write_jsonl(
        inclusion_directory / "main_cohort.jsonl", cohort_records
    )
    inclusion_manifest = {
        "decision": {
            "inventory_rule": "all supported time-bounded GAIA operational anomaly events",
            "main_cohort": MAIN_COHORT,
            "main_rule": "exclude a target event iff another root service is active at its anchor",
            "same_root_concurrency": "retained because the service-level root remains unique",
            "sensitivity_rule": "anchor has at least one concurrently active different root service",
            "target_semantics": "designated operational event root; not proof of sole causal fault in the full context window",
        },
        "files": {
            "flags.jsonl": flag_file,
            "main_cohort.jsonl": cohort_file,
        },
        "inventory_case_count": len(inputs),
        "main_case_count": len(main_ids),
        "schema_version": INCLUSION_SCHEMA_VERSION,
        "sensitivity_case_count": len(sensitivity_ids),
        "source": {
            "dataset_manifest_sha256": _sha256(manifest_directory / "manifest.json"),
            "event_audit_sha256": dataset_manifest["files"]["event_audit.jsonl"]["sha256"],
            "groups_sha256": dataset_manifest["files"]["groups.jsonl"]["sha256"],
            "selected_assignment_sha256": split_manifest["files"]["assignments.jsonl"]["sha256"],
            "split_manifest_sha256": _sha256(split_directory / "split_manifest.json"),
        },
    }
    _write_json(inclusion_directory / "manifest.json", inclusion_manifest)
    inclusion_manifest_sha = _sha256(inclusion_directory / "manifest.json")

    main_inputs = tuple(inputs_by_id[case_id] for case_id in main_ids)
    main_labels = tuple(labels_by_id[case_id] for case_id in main_ids)
    split_by_case = {case_id: assignment_by_id[case_id] for case_id in main_ids}
    folds = sorted(set(split_by_case.values()))
    output_directory = Path(args.baseline_root) / "gaia_main"

    random_rankings = predict_rankings(
        main_inputs,
        lambda case_input: deterministic_random_ranking(
            case_input, args.random_seed
        ),
    )
    random_metrics = evaluate_ranking_report(
        main_inputs, main_labels, random_rankings
    )
    random_result = _write_baseline_result(
        output_directory / "random",
        "random",
        {
            "cohort": MAIN_COHORT,
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "ranking": "sha256 per-case permutation",
            "seed": args.random_seed,
        },
        random_rankings,
        random_metrics,
        {
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "fit_scope": "none",
            "folds": [
                {
                    "fold": fold,
                    "test_case_count": sum(
                        split == fold for split in split_by_case.values()
                    ),
                }
                for fold in folds
            ],
        },
        split_by_case,
        dataset_manifest,
        manifest_directory,
        split_manifest,
        split_directory,
    )

    frequency_rankings = {}
    frequency_fold_audit = []
    for fold in folds:
        train_ids = sorted(
            case_id for case_id in main_ids if split_by_case[case_id] != fold
        )
        test_ids = sorted(
            case_id for case_id in main_ids if split_by_case[case_id] == fold
        )
        model = fit_root_frequency(
            tuple(labels_by_id[case_id] for case_id in train_ids)
        )
        fold_rankings = predict_rankings(
            tuple(inputs_by_id[case_id] for case_id in test_ids), model.rank
        )
        frequency_rankings.update(fold_rankings)
        frequency_fold_audit.append(
            {
                "fold": fold,
                "root_counts": dict(model.root_counts),
                "test_case_count": len(test_ids),
                "test_case_ids_sha256": _id_digest(test_ids),
                "train_case_count": len(train_ids),
                "train_case_ids_sha256": _id_digest(train_ids),
                "train_test_overlap": len(set(train_ids) & set(test_ids)),
            }
        )
    if set(frequency_rankings) != set(main_ids):
        raise ValueError("main-cohort frequency predictions are incomplete")
    frequency_metrics = evaluate_ranking_report(
        main_inputs, main_labels, frequency_rankings
    )
    frequency_result = _write_baseline_result(
        output_directory / "root_frequency",
        "root_frequency",
        {
            "cohort": MAIN_COHORT,
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "fit": "per-fold main-cohort root counts",
            "tie_break": "service name ascending",
        },
        frequency_rankings,
        frequency_metrics,
        {
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "fit_scope": "labels from main-cohort cases in the four training folds only",
            "folds": frequency_fold_audit,
            "tie_break": "service name ascending",
        },
        split_by_case,
        dataset_manifest,
        manifest_directory,
        split_manifest,
        split_directory,
    )

    inventory_metric_path = (
        Path(args.baseline_root) / "gaia" / "metric_change" / "predictions.jsonl"
    )
    inventory_rankings, inventory_details = _load_inventory_metric_predictions(
        inventory_metric_path
    )
    metric_rankings = {case_id: inventory_rankings[case_id] for case_id in main_ids}
    metric_details = {case_id: inventory_details[case_id] for case_id in main_ids}
    metric_predictions = {
        case_id: metric_change_prediction(
            inputs_by_id[case_id],
            {
                service: (
                    float(score) if score is not None else float("nan")
                )
                for service, score in metric_details[case_id]["service_scores"].items()
            },
            metric_details[case_id]["observed_feature_counts"],
        )
        for case_id in main_ids
    }
    metric_metrics = dict(
        evaluate_ranking_report(main_inputs, main_labels, metric_rankings)
    )
    metric_metrics["coverage_slices"] = _coverage_slice_reports(
        main_inputs,
        main_labels,
        metric_rankings,
        metric_predictions,
    )
    metric_config = json.loads(
        (
            Path(args.baseline_root)
            / "gaia"
            / "metric_change"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )["config"]
    metric_config.update(
        {
            "cohort": MAIN_COHORT,
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "source_inventory_predictions_sha256": _sha256(inventory_metric_path),
        }
    )
    metric_result = _write_baseline_result(
        output_directory / "metric_change",
        "metric_change",
        metric_config,
        metric_rankings,
        metric_metrics,
        {
            "cohort_manifest_sha256": inclusion_manifest_sha,
            "coverage": _coverage_audit(
                metric_predictions, main_inputs[0].services
            ),
            "fit_scope": "none; filtered from byte-verified inventory predictions",
            "source_inventory_predictions_sha256": _sha256(inventory_metric_path),
        },
        split_by_case,
        dataset_manifest,
        manifest_directory,
        split_manifest,
        split_directory,
        prediction_details=metric_details,
    )

    baseline_results = {
        "metric_change": _compact(
            metric_metrics, metric_result["run_manifest_sha256"]
        ),
        "random": _compact(random_metrics, random_result["run_manifest_sha256"]),
        "root_frequency": _compact(
            frequency_metrics, frequency_result["run_manifest_sha256"]
        ),
    }
    summary_path = Path(args.baseline_summary)
    baseline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if baseline_summary.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("baseline summary schema mismatch")
    baseline_summary["datasets"]["gaia_main"] = baseline_results
    baseline_summary["gaia_main_cohort"] = {
        "case_count": len(main_ids),
        "cohort": MAIN_COHORT,
        "inclusion_manifest_sha256": inclusion_manifest_sha,
        "inventory_dataset_key": "gaia",
    }
    _write_json(summary_path, baseline_summary)

    fold_counts = Counter(assignment_by_id[case_id] for case_id in main_ids)
    diagnostics = {
        "baseline_results": baseline_results,
        "cohorts": {
            "actual_interval_isolated": {
                "case_count": len(actual_isolated_ids),
                "distribution": _distribution(actual_isolated_ids, labels_by_id),
            },
            "anchor_no_concurrent_event": {
                "case_count": len(anchor_no_concurrent_ids),
                "distribution": _distribution(anchor_no_concurrent_ids, labels_by_id),
            },
            "anchor_same_root_concurrent": {
                "case_count": len(same_root_concurrent_ids),
                "distribution": _distribution(same_root_concurrent_ids, labels_by_id),
            },
            "anchor_unique_root_service_main": {
                "case_count": len(main_ids),
                "distribution": _distribution(main_ids, labels_by_id),
                "fold_case_counts": dict(sorted(fold_counts.items())),
            },
            "anchor_multi_root_sensitivity": {
                "case_count": len(sensitivity_ids),
                "distribution": _distribution(sensitivity_ids, labels_by_id),
            },
            "full_event_inventory": {
                "case_count": len(inputs),
                "distribution": _distribution(sorted(inputs_by_id), labels_by_id),
            },
        },
        "decision": inclusion_manifest["decision"],
        "inclusion_manifest_sha256": inclusion_manifest_sha,
        "schema_version": INCLUSION_SCHEMA_VERSION,
    }
    _write_json(Path(args.diagnostics_output), diagnostics)
    return diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--inclusion-output-root", required=True)
    parser.add_argument("--diagnostics-output", required=True)
    parser.add_argument("--random-seed", default=20260819, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
