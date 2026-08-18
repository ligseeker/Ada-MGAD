#!/usr/bin/env python3
"""Cross-check all P1 manifests, splits, cohorts, and baseline artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p1_sanity_baselines import _load_split
from src.data import read_manifest_cases, verify_source_snapshot
from src.evaluation import evaluate_ranking_report, validate_ranking


AUDIT_SCHEMA_VERSION = "p1_gate_audit_v1"
BASELINES = ("metric_change", "random", "root_frequency")


class GateAuditError(ValueError):
    """Raised when any P1 artifact or binding fails its gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateAuditError("cannot read {}".format(path)) from exc


def _read_jsonl(path: Path) -> tuple:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                _require(isinstance(row, dict), "{} rows must be objects".format(path))
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateAuditError("cannot read {}".format(path)) from exc
    return tuple(rows)


def _verify_indexed_file(directory: Path, filename: str, expected: Mapping) -> None:
    path = directory / filename
    _require(path.is_file(), "missing indexed file: {}".format(path))
    _require(
        _sha256(path) == expected.get("sha256"),
        "indexed checksum mismatch: {}".format(path),
    )
    if "rows" in expected:
        _require(
            len(_read_jsonl(path)) == expected["rows"],
            "indexed row mismatch: {}".format(path),
        )
    if "bytes" in expected:
        _require(
            path.stat().st_size == expected["bytes"],
            "indexed byte mismatch: {}".format(path),
        )


def _contains_sensitive_key(value) -> bool:
    if isinstance(value, dict):
        if {"fault_type", "root_service"}.intersection(value):
            return True
        return any(_contains_sensitive_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _equivalent(expected, observed) -> bool:
    """Compare reports strictly except for harmless float summation order."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        return set(expected) == set(observed) and all(
            _equivalent(expected[key], observed[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            _equivalent(left, right) for left, right in zip(expected, observed)
        )
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
    ):
        return math.isclose(float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-15)
    return expected == observed


def _verify_inclusion(
    artifact_root: Path,
    gaia_case_ids: set,
    split_by_case: Mapping[str, str],
) -> tuple:
    directory = artifact_root / "inclusion" / "gaia"
    manifest = _read_json(directory / "manifest.json")
    _require(manifest.get("schema_version") == "p1_gaia_inclusion_v1", "bad inclusion schema")
    for filename, expected in manifest["files"].items():
        _verify_indexed_file(directory, filename, expected)

    flags = _read_jsonl(directory / "flags.jsonl")
    cohort = _read_jsonl(directory / "main_cohort.jsonl")
    flag_ids = {row["case_id"] for row in flags}
    cohort_ids = {row["case_id"] for row in cohort}
    _require(len(flag_ids) == len(flags) == 16200, "GAIA flags are incomplete or duplicated")
    _require(flag_ids == gaia_case_ids, "GAIA flags do not cover the inventory")
    _require(len(cohort_ids) == len(cohort) == 13470, "GAIA main cohort count mismatch")
    _require(
        cohort_ids == {row["case_id"] for row in flags if row["main_cohort"]},
        "GAIA main cohort does not match purity flags",
    )
    _require(
        all(row["split"] == split_by_case[row["case_id"]] for row in cohort),
        "GAIA main cohort does not inherit the selected split",
    )
    _require(manifest["inventory_case_count"] == len(flags), "bad inventory count")
    _require(manifest["main_case_count"] == len(cohort), "bad main cohort count")
    _require(
        manifest["sensitivity_case_count"] == len(flags) - len(cohort),
        "bad sensitivity count",
    )
    gaia_manifest = artifact_root / "manifests" / "gaia"
    gaia_split = artifact_root / "splits" / "gaia"
    source = manifest["source"]
    expected_bindings = {
        "dataset_manifest_sha256": gaia_manifest / "manifest.json",
        "groups_sha256": gaia_manifest / "groups.jsonl",
        "selected_assignment_sha256": gaia_split / "assignments.jsonl",
        "split_manifest_sha256": gaia_split / "split_manifest.json",
    }
    for key, path in expected_bindings.items():
        _require(source[key] == _sha256(path), "stale inclusion binding: {}".format(key))
    return cohort_ids, manifest


def _verify_baseline(
    directory: Path,
    inputs: Sequence,
    labels: Sequence,
    split_by_case: Mapping[str, str],
    expected_run_sha256: str,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
) -> Mapping[str, object]:
    run = _read_json(directory / "run_manifest.json")
    _require(
        _sha256(directory / "run_manifest.json") == expected_run_sha256,
        "baseline summary has a stale run binding: {}".format(directory),
    )
    for filename, expected in run["files"].items():
        _verify_indexed_file(directory, filename, expected)
    _require(
        run["source"]["dataset_manifest_sha256"] == _sha256(dataset_manifest_path),
        "baseline dataset binding mismatch: {}".format(directory),
    )
    _require(
        run["source"]["split_manifest_sha256"] == _sha256(split_manifest_path),
        "baseline split binding mismatch: {}".format(directory),
    )
    split_manifest = _read_json(split_manifest_path)
    _require(
        run["source"]["selected_assignment_sha256"]
        == split_manifest["files"]["assignments.jsonl"]["sha256"],
        "baseline assignment binding mismatch: {}".format(directory),
    )
    _require(
        run["label_firewall"]["prediction_records_contain_root_or_fault"] is False,
        "baseline does not assert the Label Firewall: {}".format(directory),
    )

    records = _read_jsonl(directory / "predictions.jsonl")
    expected_ids = {row.case_id for row in inputs}
    observed_ids = {row["case_id"] for row in records}
    _require(len(records) == len(observed_ids), "duplicate prediction case_id")
    _require(observed_ids == expected_ids, "prediction coverage mismatch: {}".format(directory))
    _require(
        all(not _contains_sensitive_key(row) for row in records),
        "prediction contains a sensitive key: {}".format(directory),
    )
    inputs_by_id = {row.case_id: row for row in inputs}
    rankings = {}
    for record in records:
        case_id = record["case_id"]
        _require(
            record["split"] == split_by_case[case_id],
            "prediction split mismatch: {}/{}".format(directory, case_id),
        )
        _require(record["baseline"] == run["baseline"], "prediction baseline mismatch")
        rankings[case_id] = validate_ranking(inputs_by_id[case_id], record["ranking"])

    metrics = _read_json(directory / "metrics.json")
    recomputed = evaluate_ranking_report(inputs, labels, rankings)
    for section in ("fault_type", "overall", "root_service"):
        _require(
            _equivalent(metrics.get(section), recomputed[section]),
            "stored metrics do not reproduce: {}/{}".format(directory, section),
        )
    training_audit = _read_json(directory / "training_audit.json")
    if run["baseline"] == "root_frequency":
        _require(
            all(fold["train_test_overlap"] == 0 for fold in training_audit["folds"]),
            "root-frequency training/test overlap detected",
        )
    return {
        "baseline": run["baseline"],
        "case_count": len(records),
        "prediction_sha256": run["files"]["predictions.jsonl"]["sha256"],
        "run_manifest_sha256": expected_run_sha256,
    }


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    rendered = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/p1")
    parser.add_argument("--raw-source-root")
    parser.add_argument("--output", default="artifacts/p1/gate_audit.json")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    manifests = artifact_root / "manifests"
    splits = artifact_root / "splits"
    loaded = {}
    for key in ("gaia", "re2ob"):
        (
            dataset_manifest,
            split_manifest,
            inputs,
            labels,
            assignments,
        ) = _load_split(manifests / key, splits / key)
        loaded[key] = {
            "dataset_manifest": dataset_manifest,
            "split_manifest": split_manifest,
            "inputs": inputs,
            "labels": labels,
            "assignments": assignments,
        }

    gaia_inputs = loaded["gaia"]["inputs"]
    gaia_case_ids = {row.case_id for row in gaia_inputs}
    gaia_split_by_case = {
        row.case_id: row.split for row in loaded["gaia"]["assignments"]
    }
    main_ids, inclusion_manifest = _verify_inclusion(
        artifact_root, gaia_case_ids, gaia_split_by_case
    )

    source_snapshot_directory = artifact_root / "source_snapshots" / "re2ob"
    if args.raw_source_root:
        source_snapshot = verify_source_snapshot(
            str(source_snapshot_directory), args.raw_source_root
        )
        raw_source_verified = True
    else:
        source_snapshot = _read_json(source_snapshot_directory / "manifest.json")
        for filename, expected in source_snapshot["files"].items():
            _verify_indexed_file(source_snapshot_directory, filename, expected)
        raw_source_verified = False
    _require(source_snapshot["case_count"] == 90, "RE2 source snapshot case count mismatch")
    _require(
        source_snapshot["metadata"]["case_manifest_sha256"]
        == _sha256(manifests / "re2ob" / "manifest.json"),
        "RE2 source snapshot has a stale case-manifest binding",
    )

    telemetry = _read_json(artifact_root / "telemetry_diagnostics.json")
    _require(telemetry["gaia"]["scan_scope"] == "full", "GAIA scan is not full")
    _require(telemetry["re2ob"]["scan_scope"] == "full", "RE2 scan is not full")
    _require(telemetry["gaia"]["cases"] == 16200, "GAIA diagnostic count mismatch")
    _require(telemetry["re2ob"]["cases"] == 90, "RE2 diagnostic count mismatch")
    for dataset in ("gaia", "re2ob"):
        for modality in ("metrics", "logs", "traces"):
            _require(
                not telemetry[dataset]["modalities"][modality]["incomplete_files"],
                "{} {} scan is incomplete".format(dataset, modality),
            )
    total_rows = sum(
        telemetry[dataset]["modalities"][modality]["rows"]
        for dataset in ("gaia", "re2ob")
        for modality in ("metrics", "logs", "traces")
    )
    _require(total_rows == 349120558, "unexpected telemetry row total")
    re2_inventory = telemetry["re2ob"]["source_inventory"]
    consumed_index = source_snapshot["files"]["consumed_files.jsonl"]
    _require(re2_inventory["files"] == consumed_index["rows"], "RE2 source file count drift")
    _require(
        re2_inventory["bytes"] == consumed_index["indexed_bytes"],
        "RE2 source byte count drift",
    )

    inclusion_hash = _sha256(artifact_root / "inclusion" / "gaia" / "manifest.json")
    inclusion_diagnostics = _read_json(artifact_root / "gaia_inclusion_diagnostics.json")
    _require(
        inclusion_diagnostics["inclusion_manifest_sha256"] == inclusion_hash,
        "GAIA inclusion diagnostics binding is stale",
    )
    baseline_summary = _read_json(artifact_root / "baseline_summary.json")
    _require(
        baseline_summary["gaia_main_cohort"]["inclusion_manifest_sha256"]
        == inclusion_hash,
        "baseline summary has a stale GAIA cohort binding",
    )

    gaia_labels_by_id = {row.case_id: row for row in loaded["gaia"]["labels"]}
    main_inputs = tuple(row for row in gaia_inputs if row.case_id in main_ids)
    main_labels = tuple(
        gaia_labels_by_id[row.case_id] for row in main_inputs
    )
    contexts = {
        "gaia": (
            loaded["gaia"]["inputs"],
            loaded["gaia"]["labels"],
            gaia_split_by_case,
            manifests / "gaia" / "manifest.json",
            splits / "gaia" / "split_manifest.json",
        ),
        "gaia_main": (
            main_inputs,
            main_labels,
            gaia_split_by_case,
            manifests / "gaia" / "manifest.json",
            splits / "gaia" / "split_manifest.json",
        ),
        "re2ob": (
            loaded["re2ob"]["inputs"],
            loaded["re2ob"]["labels"],
            {row.case_id: row.split for row in loaded["re2ob"]["assignments"]},
            manifests / "re2ob" / "manifest.json",
            splits / "re2ob" / "split_manifest.json",
        ),
    }
    baseline_checks = []
    for dataset_key, context in contexts.items():
        inputs, labels, split_by_case, dataset_manifest_path, split_manifest_path = context
        for baseline in BASELINES:
            expected_sha = baseline_summary["datasets"][dataset_key][baseline][
                "run_manifest_sha256"
            ]
            baseline_checks.append(
                _verify_baseline(
                    artifact_root / "baselines" / dataset_key / baseline,
                    inputs,
                    labels,
                    split_by_case,
                    expected_sha,
                    dataset_manifest_path,
                    split_manifest_path,
                )
            )

    split_diagnostics = _read_json(artifact_root / "split_diagnostics.json")
    _require(split_diagnostics["protocol"]["seed"] == 20260819, "split seed drift")
    _require(
        split_diagnostics["gaia"]["selected"] == "grouped_stratified_5fold",
        "GAIA selected split drift",
    )
    _require(
        split_diagnostics["gaia"]["candidates"]["grouped_stratified_5fold"]["integrity"]
        == "pass",
        "GAIA split integrity failed",
    )
    _require(
        split_diagnostics["re2ob"]["candidate"]["integrity"] == "pass",
        "RE2 split integrity failed",
    )

    result = {
        "all_checks_passed": True,
        "baseline_outputs_verified": len(baseline_checks),
        "case_counts": {"gaia_inventory": 16200, "gaia_main": 13470, "re2ob": 90},
        "checks": {
            "G1_gaia_case_mapping_and_inclusion": "pass",
            "G2_re2ob_case_mapping_and_source_identity": "pass",
            "G3_complete_rankings": "pass",
            "G4_metrics_recomputed": "pass",
            "G5_sanity_baselines": "pass",
            "G6_split_integrity_and_bindings": "pass",
            "G7_label_firewall": "pass",
            "G8_full_telemetry_diagnostics": "pass",
        },
        "gaia_inclusion_manifest_sha256": inclusion_hash,
        "raw_re2_source_bytes_verified": raw_source_verified,
        "re2_content_identity_sha256": source_snapshot["content_identity_sha256"],
        "re2_source_snapshot_manifest_sha256": _sha256(
            source_snapshot_directory / "manifest.json"
        ),
        "schema_version": AUDIT_SCHEMA_VERSION,
        "telemetry_rows_verified": total_rows,
    }
    _require(inclusion_manifest["main_case_count"] == 13470, "GAIA main count drift")
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
