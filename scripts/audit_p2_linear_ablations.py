#!/usr/bin/env python3
"""Audit C0-L/C0-T/C1-I OOF runs and compare with C0-M."""

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_c0_metric import _read_jsonl, _sha256, _write_json
from src.data import read_manifest_cases
from src.evaluation import evaluate_ranking_report


METHODS = ("c0_l", "c0_t", "c1_i")
METRICS = ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")


def _compact(report):
    return {
        "fault_type_macro": report["fault_type"]["macro"],
        "overall": report["overall"],
        "root_service_macro": report["root_service"]["macro"],
    }


def _delta(actual, reference):
    return {
        section: {
            metric: float(actual[section][metric] - reference[section][metric])
            for metric in METRICS
        }
        for section in ("fault_type_macro", "overall", "root_service_macro")
    }


def _audit_run(method, manifest_directory, run_directory, inclusion_path):
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included = {row["case_id"] for row in _read_jsonl(inclusion_path)}
        inputs = tuple(row for row in inputs if row.case_id in included)
        labels = tuple(row for row in labels if row.case_id in included)
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    if manifest.get("run_schema_version") != "p2_nested_oof_v1":
        raise ValueError("unsupported linear ablation run schema")
    if manifest.get("method") != method:
        raise ValueError("linear ablation method mismatch")
    for filename, expected in manifest["files"].items():
        if _sha256(run_directory / filename) != expected["sha256"]:
            raise ValueError("run file checksum mismatch: {}".format(filename))

    prediction_path = run_directory / "predictions.jsonl"
    normalized = re.sub(
        r"[^a-z0-9]", "", prediction_path.read_text(encoding="utf-8").lower()
    )
    if any(
        token in normalized for token in ("faulttype", "groundtruth", "rootservice")
    ):
        raise ValueError("prediction records contain label tokens")
    predictions = _read_jsonl(prediction_path)
    expected_fields = {"case_id", "method", "ranking", "service_scores", "split"}
    if any(set(record) != expected_fields for record in predictions):
        raise ValueError("prediction record schema mismatch")
    if any(record["method"] != method for record in predictions):
        raise ValueError("prediction method mismatch")
    rankings = {row["case_id"]: tuple(row["ranking"]) for row in predictions}
    if len(rankings) != len(predictions) or set(rankings) != set(inputs_by_id):
        raise ValueError("OOF predictions are incomplete or duplicated")
    ordered_ids = sorted(inputs_by_id)
    recomputed = evaluate_ranking_report(
        tuple(inputs_by_id[case_id] for case_id in ordered_ids),
        tuple(labels_by_id[case_id] for case_id in ordered_ids),
        rankings,
    )
    recorded = json.loads((run_directory / "metrics.json").read_text())
    if recomputed != recorded:
        raise ValueError("recorded linear ablation metrics do not exactly recompute")

    training = json.loads((run_directory / "training_audit.json").read_text())
    folds = training.get("folds", ())
    if len(folds) != 5:
        raise ValueError("training audit must contain five outer folds")
    selected_cs = []
    inner_fit_count = 0
    test_digests = set()
    for fold in folds:
        if fold["train_test_overlap"] or fold["train_test_group_overlap"]:
            raise ValueError("outer fold leakage detected")
        if fold["test_case_ids_sha256"] in test_digests:
            raise ValueError("outer test digest repeated")
        test_digests.add(fold["test_case_ids_sha256"])
        selected = sorted(
            fold["inner_candidates"],
            key=lambda row: (-row["mean_root_service_macro_Avg@5"], row["C"]),
        )[0]
        if fold["selected_C"] != selected["C"]:
            raise ValueError("outer selected C does not match inner objective")
        selected_cs.append(fold["selected_C"])
        for candidate in fold["inner_candidates"]:
            if len(candidate["inner_folds"]) != 4:
                raise ValueError("candidate does not cover four inner folds")
            for inner in candidate["inner_folds"]:
                inner_fit_count += 1
                if inner["fit_validation_overlap"] or inner[
                    "fit_validation_group_overlap"
                ]:
                    raise ValueError("inner fold leakage detected")
    return {
        "audit": {
            "case_count": len(inputs),
            "core_files_verified": True,
            "inner_fit_count": inner_fit_count,
            "label_free_predictions": True,
            "metrics_exactly_recomputed": True,
            "outer_fold_count": len(folds),
            "selected_C_by_outer_fold": selected_cs,
            "test_group_overlap_max": max(
                fold["train_test_group_overlap"] for fold in folds
            ),
            "test_overlap_max": max(fold["train_test_overlap"] for fold in folds),
        },
        "metrics": _compact(recorded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--run-root", default="artifacts/p2/runs")
    parser.add_argument("--c0-m-summary", default="artifacts/p2/c0_m_summary.json")
    parser.add_argument(
        "--output", default="artifacts/p2/linear_ablation_audit.json"
    )
    args = parser.parse_args()
    manifest_root = Path(args.manifest_root)
    run_root = Path(args.run_root)
    dataset_configs = {
        "gaia_main": {
            "inclusion": Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"),
            "manifest": manifest_root / "gaia",
        },
        "re2ob": {"inclusion": None, "manifest": manifest_root / "re2ob"},
    }
    c0_m = json.loads(Path(args.c0_m_summary).read_text())["datasets"]
    results = {}
    for method in METHODS:
        results[method] = {}
        for dataset, config in dataset_configs.items():
            result = _audit_run(
                method,
                config["manifest"],
                run_root / method / dataset,
                config["inclusion"],
            )
            result["delta_vs_c0_m"] = _delta(result["metrics"], c0_m[dataset])
            results[method][dataset] = result

    best_single = {}
    for dataset in dataset_configs:
        candidates = {
            "c0_m": c0_m[dataset],
            "c0_l": results["c0_l"][dataset]["metrics"],
            "c0_t": results["c0_t"][dataset]["metrics"],
        }
        method, metrics = sorted(
            candidates.items(),
            key=lambda item: (
                -item[1]["root_service_macro"]["Avg@5"], item[0]
            ),
        )[0]
        c1_metrics = results["c1_i"][dataset]["metrics"]
        best_single[dataset] = {
            "best_single_method_by_root_macro_Avg@5": method,
            "c1_i_delta": _delta(c1_metrics, metrics),
            "c1_i_primary_improved": bool(
                c1_metrics["root_service_macro"]["Avg@5"]
                > metrics["root_service_macro"]["Avg@5"]
            ),
        }
    output = {
        "audit_schema_version": "p2_linear_ablation_audit_v1",
        "best_single_comparison": best_single,
        "c1_i_exploratory_signal_both_datasets": all(
            best_single[dataset]["c1_i_primary_improved"]
            for dataset in dataset_configs
        ),
        "methods": results,
    }
    _write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
