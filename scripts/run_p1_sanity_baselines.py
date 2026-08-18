#!/usr/bin/env python
"""Run deterministic Random and train-fold-only Frequency RCA baselines."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import deterministic_random_ranking, fit_root_frequency
from src.data import (
    CaseGroup,
    SplitAssignment,
    read_manifest_cases,
    validate_split_integrity,
    verify_manifest_bundle,
)
from src.evaluation import evaluate_ranking_report, predict_rankings


BASELINE_SCHEMA_VERSION = "p1_sanity_baseline_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(case_ids: Sequence[str]) -> str:
    payload = "".join("{}\n".format(case_id) for case_id in sorted(case_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_line(record: Mapping[str, object]) -> bytes:
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


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]):
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


def _write_json(path: Path, record: Mapping[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    rendered = json.dumps(
        record, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bytes": len(rendered.encode("utf-8")), "sha256": _sha256(path)}


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _load_split(manifest_directory: Path, split_directory: Path):
    dataset_manifest = verify_manifest_bundle(str(manifest_directory))
    split_manifest_path = split_directory / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assignment_path = split_directory / "assignments.jsonl"
    expected = split_manifest["files"]["assignments.jsonl"]
    assignment_rows = _read_jsonl(assignment_path)
    if len(assignment_rows) != expected["rows"] or _sha256(assignment_path) != expected["sha256"]:
        raise ValueError("selected split assignment checksum mismatch")

    group_rows = _read_jsonl(manifest_directory / "groups.jsonl")
    groups = tuple(
        CaseGroup(
            str(row["case_id"]),
            str(row["group_id"]),
            int(row["group_size"]),
            int(row["overlap_degree"]),
        )
        for row in group_rows
    )
    groups_by_case = {row.case_id: row.group_id for row in groups}
    assignments = tuple(
        SplitAssignment(str(row["case_id"]), str(row["split"]))
        for row in assignment_rows
    )
    for row in assignment_rows:
        if groups_by_case[str(row["case_id"])] != str(row["group_id"]):
            raise ValueError("assignment group_id does not match source manifest")
    validate_split_integrity(assignments, groups)
    inputs, labels = read_manifest_cases(str(manifest_directory))
    return dataset_manifest, split_manifest, inputs, labels, assignments


def _prediction_records(
    baseline, rankings, split_by_case, prediction_details=None
):
    records = []
    for case_id in sorted(rankings):
        record = {
            "baseline": baseline,
            "case_id": case_id,
            "ranking": list(rankings[case_id]),
            "split": split_by_case[case_id],
        }
        if prediction_details is not None:
            record.update(prediction_details[case_id])
        records.append(record)
    return tuple(records)


def _write_baseline_result(
    output_directory: Path,
    baseline: str,
    config: Mapping[str, object],
    rankings,
    metrics,
    training_audit,
    split_by_case,
    dataset_manifest,
    dataset_manifest_directory: Path,
    split_manifest,
    split_directory: Path,
    prediction_details=None,
):
    predictions = _write_jsonl(
        output_directory / "predictions.jsonl",
        _prediction_records(
            baseline, rankings, split_by_case, prediction_details
        ),
    )
    metrics_file = _write_json(output_directory / "metrics.json", metrics)
    audit_file = _write_json(
        output_directory / "training_audit.json", training_audit
    )
    run_manifest = {
        "baseline": baseline,
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "case_count": len(rankings),
        "config": dict(config),
        "dataset": dataset_manifest["dataset"],
        "files": {
            "metrics.json": metrics_file,
            "predictions.jsonl": predictions,
            "training_audit.json": audit_file,
        },
        "label_firewall": {
            "fit_scope": training_audit["fit_scope"],
            "prediction_records_contain_root_or_fault": False,
        },
        "source": {
            "dataset_manifest_path": str(
                dataset_manifest_directory / "manifest.json"
            ),
            "dataset_manifest_sha256": _sha256(
                dataset_manifest_directory / "manifest.json"
            ),
            "selected_assignment_sha256": split_manifest["files"][
                "assignments.jsonl"
            ]["sha256"],
            "split_manifest_path": str(split_directory / "split_manifest.json"),
            "split_manifest_sha256": _sha256(
                split_directory / "split_manifest.json"
            ),
        },
    }
    _write_json(output_directory / "run_manifest.json", run_manifest)
    for filename, expected in run_manifest["files"].items():
        path = output_directory / filename
        if _sha256(path) != expected["sha256"]:
            raise ValueError("baseline output checksum mismatch: {}".format(path))
    return {
        "metrics": metrics,
        "run_manifest_sha256": _sha256(output_directory / "run_manifest.json"),
    }


def _run_dataset(
    manifest_directory: Path,
    split_directory: Path,
    output_directory: Path,
    random_seed: int,
):
    (
        dataset_manifest,
        split_manifest,
        inputs,
        labels,
        assignments,
    ) = _load_split(manifest_directory, split_directory)
    split_by_case = {row.case_id: row.split for row in assignments}
    labels_by_id = {row.case_id: row for row in labels}
    inputs_by_id = {row.case_id: row for row in inputs}
    folds = sorted(set(split_by_case.values()))

    random_rankings = predict_rankings(
        inputs,
        lambda case_input: deterministic_random_ranking(case_input, random_seed),
    )
    random_metrics = evaluate_ranking_report(inputs, labels, random_rankings)
    random_audit = {
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
    }
    random_result = _write_baseline_result(
        output_directory / "random",
        "random",
        {"ranking": "sha256 per-case permutation", "seed": random_seed},
        random_rankings,
        random_metrics,
        random_audit,
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
            case_id for case_id, split in split_by_case.items() if split != fold
        )
        test_ids = sorted(
            case_id for case_id, split in split_by_case.items() if split == fold
        )
        if set(train_ids) & set(test_ids):
            raise ValueError("train/test overlap in frequency baseline")
        model = fit_root_frequency(tuple(labels_by_id[case_id] for case_id in train_ids))
        fold_rankings = predict_rankings(
            tuple(inputs_by_id[case_id] for case_id in test_ids), model.rank
        )
        if set(frequency_rankings) & set(fold_rankings):
            raise ValueError("frequency OOF predictions contain duplicate cases")
        frequency_rankings.update(fold_rankings)
        frequency_fold_audit.append(
            {
                "fold": fold,
                "root_counts": dict(model.root_counts),
                "test_case_count": len(test_ids),
                "test_case_ids_sha256": _id_digest(test_ids),
                "train_case_count": len(train_ids),
                "train_case_ids_sha256": _id_digest(train_ids),
                "train_test_overlap": 0,
            }
        )
    if set(frequency_rankings) != set(inputs_by_id):
        raise ValueError("frequency baseline did not produce complete OOF rankings")
    frequency_metrics = evaluate_ranking_report(inputs, labels, frequency_rankings)
    frequency_audit = {
        "fit_scope": "labels from the four training folds only",
        "folds": frequency_fold_audit,
        "tie_break": "service name ascending",
    }
    frequency_result = _write_baseline_result(
        output_directory / "root_frequency",
        "root_frequency",
        {"fit": "per-fold root counts", "tie_break": "service name ascending"},
        frequency_rankings,
        frequency_metrics,
        frequency_audit,
        split_by_case,
        dataset_manifest,
        manifest_directory,
        split_manifest,
        split_directory,
    )
    return {"random": random_result, "root_frequency": frequency_result}


def run_baselines(
    manifest_root: Path,
    split_root: Path,
    output_root: Path,
    summary_output: Path,
    random_seed: int,
):
    results = {
        dataset: _run_dataset(
            manifest_root / dataset,
            split_root / dataset,
            output_root / dataset,
            random_seed,
        )
        for dataset in ("gaia", "re2ob")
    }
    if summary_output.exists():
        summary = json.loads(summary_output.read_text(encoding="utf-8"))
        if summary.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
            raise ValueError("existing baseline summary uses another schema")
    else:
        summary = {
            "baseline_schema_version": BASELINE_SCHEMA_VERSION,
            "datasets": {"gaia": {}, "re2ob": {}},
        }
    for dataset, baselines in results.items():
        dataset_summary = summary["datasets"].setdefault(dataset, {})
        for baseline, result in baselines.items():
            dataset_summary[baseline] = {
                "fault_type_macro": result["metrics"]["fault_type"]["macro"],
                "overall": result["metrics"]["overall"],
                "root_service_macro": result["metrics"]["root_service"]["macro"],
                "run_manifest_sha256": result["run_manifest_sha256"],
            }
    summary["random_seed"] = random_seed
    _write_json(summary_output, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--random-seed", default=20260819, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_baselines(
        Path(args.manifest_root),
        Path(args.split_root),
        Path(args.output_root),
        Path(args.summary_output),
        args.random_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
