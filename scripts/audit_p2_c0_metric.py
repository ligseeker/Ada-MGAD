#!/usr/bin/env python3
"""Independently audit C0-M OOF outputs and compare them with P1 B2."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_manifest_cases
from src.evaluation import evaluate_ranking_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _write_json(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        record, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _nested_delta(actual, reference):
    return {
        section: {
            metric: float(actual[section][metric] - reference[section][metric])
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
        }
        for section in ("fault_type_macro", "overall", "root_service_macro")
    }


def _audit_dataset(
    dataset_key,
    manifest_directory,
    run_directory,
    inclusion_path,
    p1_reference,
):
    inputs, labels = read_manifest_cases(str(manifest_directory))
    if inclusion_path is not None:
        included = {row["case_id"] for row in _read_jsonl(inclusion_path)}
        inputs = tuple(row for row in inputs if row.case_id in included)
        labels = tuple(row for row in labels if row.case_id in included)
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}

    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    if manifest.get("run_schema_version") != "p2_nested_oof_v1":
        raise ValueError("unsupported C0-M run schema")
    for filename, expected in manifest["files"].items():
        if _sha256(run_directory / filename) != expected["sha256"]:
            raise ValueError("run file checksum mismatch: {}".format(filename))

    prediction_path = run_directory / "predictions.jsonl"
    prediction_text = prediction_path.read_text(encoding="utf-8")
    normalized = re.sub(r"[^a-z0-9]", "", prediction_text.lower())
    sensitive = [
        token
        for token in ("faulttype", "groundtruth", "rootservice")
        if token in normalized
    ]
    if sensitive:
        raise ValueError("prediction records contain label tokens")
    predictions = _read_jsonl(prediction_path)
    if any(
        set(record) != {"case_id", "method", "ranking", "service_scores", "split"}
        for record in predictions
    ):
        raise ValueError("prediction record schema mismatch")
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
        raise ValueError("recorded C0-M metrics do not exactly recompute")

    audit = json.loads((run_directory / "training_audit.json").read_text())
    folds = audit.get("folds", ())
    if len(folds) != 5:
        raise ValueError("training audit must contain five outer folds")
    test_digests = set()
    selected_cs = []
    inner_fit_count = 0
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
                if (
                    inner["fit_validation_overlap"]
                    or inner["fit_validation_group_overlap"]
                ):
                    raise ValueError("inner fold leakage detected")

    compact = {
        "fault_type_macro": recorded["fault_type"]["macro"],
        "overall": recorded["overall"],
        "root_service_macro": recorded["root_service"]["macro"],
    }
    return {
        "case_count": len(inputs),
        "core_files_verified": True,
        "inner_fit_count": inner_fit_count,
        "label_free_predictions": True,
        "metrics_exactly_recomputed": True,
        "outer_fold_count": len(folds),
        "p1_b2_delta": _nested_delta(compact, p1_reference),
        "selected_C_by_outer_fold": selected_cs,
        "test_group_overlap_max": max(
            fold["train_test_group_overlap"] for fold in folds
        ),
        "test_overlap_max": max(fold["train_test_overlap"] for fold in folds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--run-root", default="artifacts/p2/runs/c0_m")
    parser.add_argument("--p1-summary", default="artifacts/p1/baseline_summary.json")
    parser.add_argument("--output", default="artifacts/p2/c0_m_audit.json")
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    run_root = Path(args.run_root)
    p1 = json.loads(Path(args.p1_summary).read_text())["datasets"]
    result = {
        "gaia_main": _audit_dataset(
            "gaia_main",
            manifest_root / "gaia",
            run_root / "gaia_main",
            Path("artifacts/p1/inclusion/gaia/main_cohort.jsonl"),
            p1["gaia_main"]["metric_change"],
        ),
        "re2ob": _audit_dataset(
            "re2ob",
            manifest_root / "re2ob",
            run_root / "re2ob",
            None,
            p1["re2ob"]["metric_change"],
        ),
    }
    result["audit_schema_version"] = "p2_c0_m_audit_v1"
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
