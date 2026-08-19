#!/usr/bin/env python3
"""Run the frozen paired bootstrap for M1-S versus C1-I (hypothesis H1)."""

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


BOOTSTRAP_SCHEMA_VERSION = "p2_m1_s_paired_bootstrap_v1"
METHOD = "m1_s"
COMPARATOR = "c1_i"
COMPARISON = "m1_s minus c1_i under the frozen nested OOF protocol"
HYPOTHESIS = "H1"
PRIMARY_METRIC = "Avg@5"
SECONDARY_METRIC = "AC@1"
SECONDARY_GUARDRAIL = -0.01


def _prediction_metrics(predictions, labels_by_id):
    values = {SECONDARY_METRIC: [], PRIMARY_METRIC: []}
    for case_id in sorted(labels_by_id):
        ranking = tuple(predictions[case_id])
        root = labels_by_id[case_id].root_service
        values[SECONDARY_METRIC].append(hit_at_k(ranking, root, 1))
        values[PRIMARY_METRIC].append(average_at_k(ranking, root, 5))
    return values


def _load_rankings(path):
    rows = _read_jsonl(path)
    rankings = {row["case_id"]: tuple(row["ranking"]) for row in rows}
    if len(rankings) != len(rows):
        raise ValueError("bootstrap predictions contain duplicate case IDs")
    return rankings


def decide_gate(results):
    """Apply the pre-registered P2-G4 go/no-go rule to both datasets."""

    if not results:
        raise ValueError("gate decision requires at least one dataset")
    exploratory = all(
        result["metric_results"][PRIMARY_METRIC]["point_delta"] > 0
        for result in results.values()
    )
    primary_ci = all(
        result["metric_results"][PRIMARY_METRIC]["ci95_lower"] > 0
        for result in results.values()
    )
    secondary_guardrail = all(
        result["metric_results"][SECONDARY_METRIC]["point_delta"]
        >= SECONDARY_GUARDRAIL
        for result in results.values()
    )
    claim_ready = bool(primary_ci and secondary_guardrail)
    if claim_ready:
        decision = "claim-ready"
    elif exploratory:
        decision = "exploratory-signal-only"
    else:
        decision = "no-go"
    return {
        "claim_ready": claim_ready,
        "claim_ready_checks": {
            "both_primary_ci_lower_bounds_positive": primary_ci,
            "both_secondary_point_deltas_at_least_minus_0_01": secondary_guardrail,
        },
        "exploratory_signal": exploratory,
        "p2_g4_decision": decision,
    }


def _run_dataset(
    dataset,
    manifest_directory,
    split_directory,
    run_root,
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
    actual = _load_rankings(run_root / METHOD / dataset / "predictions.jsonl")
    reference = _load_rankings(run_root / COMPARATOR / dataset / "predictions.jsonl")
    if set(actual) != set(case_ids) or set(reference) != set(case_ids):
        raise ValueError("bootstrap predictions do not cover cohort")
    result = paired_root_macro_bootstrap(
        case_ids,
        tuple(labels_by_id[case_id].root_service for case_id in case_ids),
        tuple(
            assignments[case_id]["group_id"] if dataset == "gaia_main" else case_id
            for case_id in case_ids
        ),
        _prediction_metrics(actual, labels_by_id),
        _prediction_metrics(reference, labels_by_id),
        iterations=iterations,
        random_seed=random_seed,
    )
    result.update(
        {
            "case_count": len(case_ids),
            "comparator": COMPARATOR,
            "comparison": COMPARISON,
            "resampling_unit": "context_group" if dataset == "gaia_main" else "case",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--run-root", default="artifacts/p2/runs")
    parser.add_argument("--stage-audit", default="artifacts/p2/m1_s_audit.json")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--output", default="artifacts/p2/m1_s_bootstrap.json")
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
            Path(args.stage_audit),
            run_root / METHOD,
            run_root / COMPARATOR,
        )
        if not path.exists()
    )
    if missing:
        raise FileNotFoundError(
            "required source bindings are missing: {}".format(", ".join(missing))
        )
    audit = json.loads(Path(args.stage_audit).read_text(encoding="utf-8"))
    if audit.get("comparator") != COMPARATOR:
        raise ValueError("stage audit does not compare M1-S against C1-I")

    results = {}
    for dataset, config in configs.items():
        results[dataset] = _run_dataset(
            dataset,
            config["manifest"],
            config["split"],
            run_root,
            config["inclusion"],
            args.iterations,
            args.random_seed,
        )
        expected_delta = audit["c1_i_comparison"][dataset]["m1_s_delta"][
            "root_service_macro"
        ]
        for metric in (SECONDARY_METRIC, PRIMARY_METRIC):
            actual_delta = results[dataset]["metric_results"][metric]["point_delta"]
            if abs(actual_delta - expected_delta[metric]) > 1e-12:
                raise ValueError("bootstrap point estimate does not match OOF report")

    output = {
        "bootstrap_schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "hypothesis": HYPOTHESIS,
        "method": METHOD,
        "primary_endpoint": "root_service_macro Avg@5",
        "results": results,
        "secondary_endpoint": "root_service_macro AC@1",
        **decide_gate(results),
    }
    _write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
