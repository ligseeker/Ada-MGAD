#!/usr/bin/env python3
"""Run the frozen paired bootstrap for C1-I versus the best single modality."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_c0_metric import _read_jsonl, _write_json
from src.data import read_manifest_cases
from src.evaluation import average_at_k, hit_at_k, paired_root_macro_bootstrap


def _prediction_metrics(predictions, labels_by_id):
    values = {"AC@1": [], "Avg@5": []}
    for case_id in sorted(labels_by_id):
        ranking = tuple(predictions[case_id])
        root = labels_by_id[case_id].root_service
        values["AC@1"].append(hit_at_k(ranking, root, 1))
        values["Avg@5"].append(average_at_k(ranking, root, 5))
    return values


def _load_rankings(path):
    rows = _read_jsonl(path)
    rankings = {row["case_id"]: tuple(row["ranking"]) for row in rows}
    if len(rankings) != len(rows):
        raise ValueError("bootstrap predictions contain duplicate case IDs")
    return rankings


def _run_dataset(
    dataset,
    manifest_directory,
    split_directory,
    run_root,
    comparator,
    inclusion_path,
    iterations,
    random_seed,
):
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included = {row["case_id"] for row in _read_jsonl(inclusion_path)}
        inputs = tuple(row for row in inputs if row.case_id in included)
        labels = tuple(row for row in labels if row.case_id in included)
    labels_by_id = {row.case_id: row for row in labels}
    case_ids = tuple(sorted(labels_by_id))
    assignments = {
        row["case_id"]: row
        for row in _read_jsonl(split_directory / "assignments.jsonl")
        if row["case_id"] in labels_by_id
    }
    if set(assignments) != set(case_ids):
        raise ValueError("bootstrap split assignments do not cover cohort")
    actual = _load_rankings(run_root / "c1_i" / dataset / "predictions.jsonl")
    reference = _load_rankings(
        run_root / comparator / dataset / "predictions.jsonl"
    )
    if set(actual) != set(case_ids) or set(reference) != set(case_ids):
        raise ValueError("bootstrap predictions do not cover cohort")
    actual_metrics = _prediction_metrics(actual, labels_by_id)
    reference_metrics = _prediction_metrics(reference, labels_by_id)
    result = paired_root_macro_bootstrap(
        case_ids,
        tuple(labels_by_id[case_id].root_service for case_id in case_ids),
        tuple(
            assignments[case_id]["group_id"] if dataset == "gaia_main" else case_id
            for case_id in case_ids
        ),
        actual_metrics,
        reference_metrics,
        iterations=iterations,
        random_seed=random_seed,
    )
    result.update(
        {
            "case_count": len(case_ids),
            "comparator": comparator,
            "comparison": "c1_i minus best single modality chosen by root-macro Avg@5 point estimate",
            "resampling_unit": "context_group" if dataset == "gaia_main" else "case",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--run-root", default="artifacts/p2/runs")
    parser.add_argument(
        "--linear-audit", default="artifacts/p2/linear_ablation_audit.json"
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--output", default="artifacts/p2/c1_i_bootstrap.json")
    args = parser.parse_args()
    audit = json.loads(Path(args.linear_audit).read_text())
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
    results = {}
    for dataset, config in configs.items():
        comparator = audit["best_single_comparison"][dataset][
            "best_single_method_by_root_macro_Avg@5"
        ]
        results[dataset] = _run_dataset(
            dataset,
            config["manifest"],
            config["split"],
            run_root,
            comparator,
            config["inclusion"],
            args.iterations,
            args.random_seed,
        )
        expected_delta = audit["best_single_comparison"][dataset]["c1_i_delta"][
            "root_service_macro"
        ]
        for metric in ("AC@1", "Avg@5"):
            actual_delta = results[dataset]["metric_results"][metric]["point_delta"]
            if abs(actual_delta - expected_delta[metric]) > 1e-12:
                raise ValueError("bootstrap point estimate does not match OOF report")
    exploratory = all(
        results[dataset]["metric_results"]["Avg@5"]["point_delta"] > 0
        for dataset in configs
    )
    primary_ci = all(
        results[dataset]["metric_results"]["Avg@5"]["ci95_lower"] > 0
        for dataset in configs
    )
    secondary_guardrail = all(
        results[dataset]["metric_results"]["AC@1"]["point_delta"] >= -0.01
        for dataset in configs
    )
    output = {
        "bootstrap_schema_version": "p2_c1_i_paired_bootstrap_v1",
        "claim_ready": bool(primary_ci and secondary_guardrail),
        "claim_ready_checks": {
            "both_primary_ci_lower_bounds_positive": primary_ci,
            "both_secondary_point_deltas_at_least_minus_0_01": secondary_guardrail,
        },
        "exploratory_signal": exploratory,
        "results": results,
    }
    _write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
