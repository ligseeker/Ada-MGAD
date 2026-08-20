#!/usr/bin/env python
"""E5: independent gate audit for the RE2-TT protocol extension.

Recomputes E1-E4 from the artifacts rather than trusting the run scripts' own
reports: file digests, label-freeness, candidate-space identity, fold isolation,
every ranking metric at all three layers, and the pre-registered headroom gate.
Nothing here imports the E1/E3/E4 drivers' computation helpers, so a bug in one of
them cannot hide itself in the audit.

This is also the only extension stage allowed to read labels while touching raw
telemetry, so it owns the root-conditioned coverage facts that
``telemetry_diagnostics.json`` deliberately excludes: for each case, whether the
labelled root service actually emits metrics, logs, and traces on both sides of the
frozen [-300 s, +300 s) window. That number decides whether a modality can carry a
root signal at all, and it cannot be computed label-free.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import (
    assert_label_free,
    read_manifest_cases,
    validate_split_integrity,
    verify_manifest_bundle,
    SplitAssignment,
    CaseGroup,
)
from src.evaluation import evaluate_ranking_report, validate_ranking

from scripts.prepare_ext_re2tt_manifests import _assert_isolated


DEFAULT_ROOT = "artifacts/ext/re2tt"
FROZEN_RE2OB_MANIFEST = "artifacts/p1/manifests/re2ob"
DEFAULT_OUTPUT = "artifacts/ext/re2tt/gate_audit.json"

AUDIT_SCHEMA_VERSION = "ext_re2tt_gate_audit_v1"
EXPECTED_CASES = 90
EXPECTED_CANDIDATES = 68
EXPECTED_FOLDS = 5
EXPECTED_SEED = 20260819
WINDOW_MS = 300_000
METRIC_TOLERANCE = 1e-12
SATURATION_CEILING = 0.90
SELECTION_THRESHOLDS = {
    "max_case_count_relative_deviation": 0.15,
    "max_fault_type_total_variation": 0.10,
    "max_root_service_total_variation": 0.10,
}
LOG_TIMESTAMP_SCALE_MS = 1e-6
TRACE_TIMESTAMP_SCALE_MS = 1e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _total_variation(observed: Mapping[str, int], reference: Mapping[str, float]) -> float:
    total = sum(observed.values())
    if total == 0:
        return 1.0
    keys = set(observed) | set(reference)
    return 0.5 * sum(
        abs(observed.get(key, 0) / total - reference.get(key, 0.0)) for key in keys
    )


def _audit_manifests(manifest_directory: Path, frozen_re2ob: Path):
    verify_manifest_bundle(str(manifest_directory))
    manifest = _read_json(manifest_directory / "manifest.json")
    recomputed = {
        name: _sha256(manifest_directory / name) for name in sorted(manifest["files"])
    }
    digest_matches = {
        name: recomputed[name] == manifest["files"][name]["sha256"]
        for name in sorted(recomputed)
    }

    inputs, labels = read_manifest_cases(str(manifest_directory))
    for case_input in inputs:
        assert_label_free(case_input)

    candidate_tuples = {case_input.services for case_input in inputs}
    labels_by_id = {row.case_id: row for row in labels}
    roots_outside = sorted(
        row.case_id
        for row in labels
        if row.root_service not in dict.fromkeys(
            next(
                case_input.services
                for case_input in inputs
                if case_input.case_id == row.case_id
            )
        )
    )

    raw_text = (manifest_directory / "inputs.jsonl").read_text(encoding="utf-8")
    leakage = {
        "condition_directory_names": "_cpu/" in raw_text or "_socket/" in raw_text,
        "raw_home_paths": "/home/" in raw_text,
        "release_directory": "RE2-TT/" in raw_text,
    }

    collisions = 0
    if frozen_re2ob.is_dir():
        frozen_ids = {
            row["case_id"] for row in _read_jsonl(frozen_re2ob / "inputs.jsonl")
        }
        collisions = len(frozen_ids & {row.case_id for row in inputs})

    return {
        "candidate_service_count": len(next(iter(candidate_tuples))),
        "candidate_tuple_variants": len(candidate_tuples),
        "case_count": len(inputs),
        "dataset": manifest["dataset"],
        "digest_matches": digest_matches,
        "excluded_case_count": manifest["metadata"]["excluded_cases"],
        "fault_type_counts": dict(
            sorted(Counter(row.fault_type for row in labels).items())
        ),
        "frozen_re2ob_case_id_collisions": collisions,
        "label_free_inputs": len(inputs),
        "manifest_schema_version": manifest["schema_version"],
        "raw_path_leakage": leakage,
        "root_service_counts": dict(
            sorted(Counter(row.root_service for row in labels).items())
        ),
        "roots_outside_candidate_space": roots_outside,
        "passed": (
            all(digest_matches.values())
            and len(inputs) == EXPECTED_CASES
            and len(candidate_tuples) == 1
            and len(next(iter(candidate_tuples))) == EXPECTED_CANDIDATES
            and not roots_outside
            and not any(leakage.values())
            and collisions == 0
            and manifest["metadata"]["excluded_cases"] == 0
        ),
    }, inputs, labels_by_id


def _audit_snapshot(snapshot_directory: Path):
    if not snapshot_directory.is_dir():
        return {"status": "absent", "passed": False}
    index = _read_json(snapshot_directory / "manifest.json")
    recomputed = {
        name: _sha256(snapshot_directory / name) for name in sorted(index["files"])
    }
    matches = {
        name: recomputed[name] == index["files"][name]["sha256"]
        for name in sorted(recomputed)
    }
    return {
        "archive_count": index["files"]["archives.jsonl"]["rows"],
        "bytes_by_role": index["files"]["consumed_files.jsonl"]["bytes_by_role"],
        "case_count": index["case_count"],
        "consumed_bytes": index["files"]["consumed_files.jsonl"]["indexed_bytes"],
        "consumed_file_count": index["files"]["consumed_files.jsonl"]["rows"],
        "content_identity_sha256": index["content_identity_sha256"],
        "digest_matches": matches,
        "passed": all(matches.values()) and index["case_count"] == EXPECTED_CASES,
        "schema_version": index["schema_version"],
        "source_root_name": index["source_root_name"],
        "status": "present",
    }


def _audit_splits(split_directory: Path, manifest_directory: Path, labels_by_id):
    manifest = _read_json(split_directory / "split_manifest.json")
    rows = _read_jsonl(split_directory / "assignments.jsonl")
    groups_rows = _read_jsonl(manifest_directory / "groups.jsonl")

    digest_matches = {
        name: _sha256(split_directory / name) == spec["sha256"]
        for name, spec in sorted(manifest["files"].items())
    }
    assignments = tuple(
        SplitAssignment(case_id=row["case_id"], split=row["split"]) for row in rows
    )
    groups = tuple(
        CaseGroup(
            case_id=row["case_id"],
            group_id=row["group_id"],
            group_size=row["group_size"],
            overlap_degree=row["overlap_degree"],
        )
        for row in groups_rows
    )
    validate_split_integrity(assignments, groups)

    group_by_case = {row.case_id: row.group_id for row in groups}
    by_fold = defaultdict(set)
    for row in rows:
        by_fold[row["fold"]].add(row["case_id"])
    folds = sorted(by_fold)
    # the int fold and the "fold-<n>" label are written independently; disagreement
    # would make every downstream fold statistic ambiguous
    fold_label_consistent = all(
        row["split"] == "fold-{}".format(row["fold"]) for row in rows
    )

    case_overlaps = 0
    group_overlaps = 0
    for left in folds:
        for right in folds:
            if left >= right:
                continue
            case_overlaps += len(by_fold[left] & by_fold[right])
            group_overlaps += len(
                {group_by_case[case] for case in by_fold[left]}
                & {group_by_case[case] for case in by_fold[right]}
            )

    total = len(assignments)
    root_reference = {
        root: count / total
        for root, count in Counter(
            row.root_service for row in labels_by_id.values()
        ).items()
    }
    fault_reference = {
        fault: count / total
        for fault, count in Counter(
            row.fault_type for row in labels_by_id.values()
        ).items()
    }
    fold_rows = []
    for fold in folds:
        cases = by_fold[fold]
        roots = Counter(labels_by_id[case].root_service for case in cases)
        faults = Counter(labels_by_id[case].fault_type for case in cases)
        fold_rows.append(
            {
                "case_count": len(cases),
                "fault_type_counts": dict(sorted(faults.items())),
                "fault_type_total_variation": _total_variation(faults, fault_reference),
                "fold": fold,
                "root_service_counts": dict(sorted(roots.items())),
                "root_service_total_variation": _total_variation(roots, root_reference),
            }
        )

    expected = total / len(folds)
    max_deviation = max(
        abs(row["case_count"] - expected) / expected for row in fold_rows
    )
    max_fault_tv = max(row["fault_type_total_variation"] for row in fold_rows)
    max_root_tv = max(row["root_service_total_variation"] for row in fold_rows)
    every_root = all(
        set(row["root_service_counts"]) == set(root_reference) for row in fold_rows
    )
    every_fault = all(
        set(row["fault_type_counts"]) == set(fault_reference) for row in fold_rows
    )

    return {
        "algorithm": manifest["algorithm"],
        "case_overlaps_between_folds": case_overlaps,
        "digest_matches": digest_matches,
        "every_fault_type_in_every_fold": every_fault,
        "every_root_service_in_every_fold": every_root,
        "fold_count": len(folds),
        "fold_label_consistent": fold_label_consistent,
        "folds": fold_rows,
        "group_overlaps_between_folds": group_overlaps,
        "max_case_count_relative_deviation": max_deviation,
        "max_fault_type_total_variation": max_fault_tv,
        "max_root_service_total_variation": max_root_tv,
        "seed": manifest["seed"],
        "selection_thresholds": SELECTION_THRESHOLDS,
        "split_schema_version": manifest["split_schema_version"],
        "passed": (
            all(digest_matches.values())
            and len(folds) == EXPECTED_FOLDS
            and manifest["seed"] == EXPECTED_SEED
            and fold_label_consistent
            and case_overlaps == 0
            and group_overlaps == 0
            and every_root
            and every_fault
            and max_deviation <= SELECTION_THRESHOLDS["max_case_count_relative_deviation"]
            and max_fault_tv <= SELECTION_THRESHOLDS["max_fault_type_total_variation"]
            and max_root_tv <= SELECTION_THRESHOLDS["max_root_service_total_variation"]
        ),
    }


def _audit_baselines(baseline_root: Path, inputs, labels_by_id):
    inputs_by_id = {case_input.case_id: case_input for case_input in inputs}
    labels = tuple(labels_by_id[case_input.case_id] for case_input in inputs)
    rows = {}
    for directory in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
        baseline = directory.name
        recorded = _read_json(directory / "metrics.json")
        predictions = _read_jsonl(directory / "predictions.jsonl")
        rankings = {}
        invalid = []
        for record in predictions:
            case_input = inputs_by_id[record["case_id"]]
            try:
                rankings[record["case_id"]] = validate_ranking(
                    case_input, record["ranking"]
                )
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                invalid.append({"case_id": record["case_id"], "error": str(error)})
        recomputed = evaluate_ranking_report(inputs, labels, rankings)

        deltas = {}
        for layer in ("overall", "fault_type", "root_service"):
            recorded_layer = (
                recorded[layer] if layer == "overall" else recorded[layer]["macro"]
            )
            recomputed_layer = (
                recomputed[layer] if layer == "overall" else recomputed[layer]["macro"]
            )
            deltas[layer] = {
                metric: abs(recomputed_layer[metric] - recorded_layer[metric])
                for metric in sorted(recorded_layer)
            }
        worst = max(
            value for layer in deltas.values() for value in layer.values()
        )
        rows[baseline] = {
            "complete_oof_coverage": set(rankings) == set(inputs_by_id),
            "invalid_rankings": invalid,
            "max_absolute_metric_delta": worst,
            "prediction_count": len(predictions),
            "recomputed_root_service_macro": dict(
                sorted(recomputed["root_service"]["macro"].items())
            ),
            "run_manifest_digest_matches": _sha256(directory / "run_manifest.json")
            is not None
            and all(
                _sha256(directory / name) == spec["sha256"]
                for name, spec in sorted(
                    _read_json(directory / "run_manifest.json")["files"].items()
                )
            ),
            "passed": (
                not invalid
                and set(rankings) == set(inputs_by_id)
                and worst <= METRIC_TOLERANCE
                and len(predictions) == EXPECTED_CASES
            ),
        }
    return {
        "baselines": rows,
        "metric_tolerance": METRIC_TOLERANCE,
        "passed": all(row["passed"] for row in rows.values()),
    }


def _audit_headroom(baselines, recorded_gate_path: Path):
    b1 = baselines["baselines"]["root_frequency"]["recomputed_root_service_macro"]
    b2 = baselines["baselines"]["metric_change"]["recomputed_root_service_macro"]
    checks = {
        "H-1": {
            "criterion": "B2 root-macro Avg@5 <= {}".format(SATURATION_CEILING),
            "kind": "gate",
            "observed": b2["Avg@5"],
            "passed": b2["Avg@5"] <= SATURATION_CEILING,
        },
        "H-2": {
            "criterion": "B2 root-macro AC@1 <= {}".format(SATURATION_CEILING),
            "kind": "gate",
            "observed": b2["AC@1"],
            "passed": b2["AC@1"] <= SATURATION_CEILING,
        },
        "H-3": {
            "criterion": "B2 strictly beats B1 on the primary and key secondary endpoint",
            "kind": "gate",
            "observed": {
                "AC@1": b2["AC@1"] - b1["AC@1"],
                "Avg@5": b2["Avg@5"] - b1["Avg@5"],
            },
            "passed": b2["Avg@5"] > b1["Avg@5"] and b2["AC@1"] > b1["AC@1"],
        },
        "H-4": {
            "criterion": "B1 root-macro AC@5 (prior degeneracy witness)",
            "kind": "report_only",
            "observed": b1["AC@5"],
            "passed": None,
        },
    }
    gates = {name: row for name, row in checks.items() if row["kind"] == "gate"}
    decision = "pass" if all(row["passed"] for row in gates.values()) else "fail"
    agreement = None
    if recorded_gate_path.is_file():
        recorded = _read_json(recorded_gate_path)["gate"]
        agreement = recorded["decision"] == decision and sorted(
            recorded["failed_gates"]
        ) == sorted(name for name, row in gates.items() if not row["passed"])
    return {
        "checks": checks,
        "decision": decision,
        "failed_gates": sorted(name for name, row in gates.items() if not row["passed"]),
        "headroom_above_b2_primary": 1.0 - b2["Avg@5"],
        "matches_recorded_gate": agreement,
        "saturation_ceiling": SATURATION_CEILING,
    }


def _root_window_presence(source_row, root_service: str, chunk_rows: int):
    """Presence of the labelled root in [t0-300s, t0) and [t0, t0+300s)."""

    anchor_ms = int(float(Path(source_row["inject_time_path"]).read_text().strip()) * 1000)
    low = anchor_ms - WINDOW_MS
    high = anchor_ms + WINDOW_MS
    presence = {}

    frame = pd.read_csv(source_row["metrics_path"], nrows=0)
    prefix = root_service + "_"
    columns = [name for name in frame.columns if name.startswith(prefix)]
    if columns:
        pre = post = 0
        for chunk in pd.read_csv(
            source_row["metrics_path"],
            usecols=["time"] + columns,
            chunksize=chunk_rows,
        ):
            stamps = np.floor(
                pd.to_numeric(chunk["time"], errors="coerce").to_numpy(dtype=np.float64)
                * 1000.0
            )
            # a scheduled row with every root column empty carries no signal
            finite = (
                chunk[columns]
                .apply(pd.to_numeric, errors="coerce")
                .notna()
                .any(axis=1)
                .to_numpy()
            )
            pre += int(np.count_nonzero((stamps >= low) & (stamps < anchor_ms) & finite))
            post += int(np.count_nonzero((stamps >= anchor_ms) & (stamps < high) & finite))
        presence["metrics"] = {"post_rows": post, "pre_rows": pre}
    else:
        presence["metrics"] = {"post_rows": 0, "pre_rows": 0}

    for modality, path_key, time_column, name_column, scale in (
        ("logs", "logs_path", "timestamp", "container_name", LOG_TIMESTAMP_SCALE_MS),
        ("traces", "traces_path", "startTime", "serviceName", TRACE_TIMESTAMP_SCALE_MS),
    ):
        pre = post = 0
        for chunk in pd.read_csv(
            source_row[path_key],
            usecols=[time_column, name_column],
            chunksize=chunk_rows,
        ):
            selected = chunk[chunk[name_column].astype(str) == root_service]
            if selected.empty:
                continue
            stamps = (
                pd.to_numeric(selected[time_column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                * scale
            )
            pre += int(np.count_nonzero((stamps >= low) & (stamps < anchor_ms)))
            post += int(np.count_nonzero((stamps >= anchor_ms) & (stamps < high)))
        presence[modality] = {"post_rows": post, "pre_rows": pre}
    return presence


def _audit_root_coverage(manifest_directory: Path, labels_by_id, chunk_rows: int, progress_every: int):
    sources = {
        row["case_id"]: row for row in _read_jsonl(manifest_directory / "sources.jsonl")
    }
    rows = []
    both_sides = Counter()
    any_side = Counter()
    for index, case_id in enumerate(sorted(sources), start=1):
        root = labels_by_id[case_id].root_service
        presence = _root_window_presence(sources[case_id], root, chunk_rows)
        record = {"case_id": case_id, "root_service": root}
        for modality, counts in sorted(presence.items()):
            covered_both = counts["pre_rows"] > 0 and counts["post_rows"] > 0
            record[modality] = {
                "covered_both_sides": covered_both,
                "post_rows": counts["post_rows"],
                "pre_rows": counts["pre_rows"],
            }
            both_sides[modality] += int(covered_both)
            any_side[modality] += int(
                counts["pre_rows"] > 0 or counts["post_rows"] > 0
            )
        rows.append(record)
        if progress_every > 0 and (index % progress_every == 0 or index == len(sources)):
            print(
                "[gate-audit] root coverage {}/{}".format(index, len(sources)),
                file=sys.stderr,
                flush=True,
            )

    gaps = {}
    for modality in sorted(both_sides):
        missing = sorted(
            row["case_id"] for row in rows if not row[modality]["covered_both_sides"]
        )
        gaps[modality] = {
            "cases_covered_both_sides": both_sides[modality],
            "cases_with_any_activity": any_side[modality],
            "cases_without_both_sides": len(missing),
            "roots_without_both_sides": dict(
                sorted(
                    Counter(
                        row["root_service"]
                        for row in rows
                        if not row[modality]["covered_both_sides"]
                    ).items()
                )
            ),
            "case_ids_without_both_sides": missing,
        }
    return {
        "cases": len(rows),
        "definition": (
            "the labelled root service emits at least one row in both [t0-300s, t0) "
            "and [t0, t0+300s); metrics additionally require a finite value cell"
        ),
        "modalities": gaps,
        "note": (
            "a modality that cannot see the root on both sides of the anchor cannot "
            "carry a comparison-block signal for that case; the frozen rule masks it"
        ),
        "per_case": rows,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", default=DEFAULT_ROOT)
    parser.add_argument("--frozen-re2ob-manifest", default=FROZEN_RE2OB_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", default=500_000, type=int)
    parser.add_argument("--progress-every", default=10, type=int)
    parser.add_argument(
        "--skip-root-coverage",
        action="store_true",
        help="skip the raw-telemetry rescan; the E1-E4 re-checks still run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.extension_root)
    output = Path(args.output)
    _assert_isolated(output.parent)

    manifests, inputs, labels_by_id = _audit_manifests(
        root / "manifests", Path(args.frozen_re2ob_manifest)
    )
    splits = _audit_splits(root / "splits", root / "manifests", labels_by_id)
    snapshot = _audit_snapshot(root / "source_snapshot")
    baselines = _audit_baselines(root / "baselines", inputs, labels_by_id)
    headroom = _audit_headroom(baselines, root / "headroom_gate.json")

    sections = {
        "e1_manifests": manifests,
        "e1_source_snapshot": snapshot,
        "e3_splits": splits,
        "e4_baselines": baselines,
    }
    audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "extension": "re2tt_protocol_extension",
        "headroom_gate": headroom,
        "independence": (
            "every metric, digest, and fold statistic is recomputed here from the "
            "artifacts; no computation helper is imported from the run scripts"
        ),
        "integrity_gates_passed": all(section["passed"] for section in sections.values()),
        "sections": sections,
    }
    if not args.skip_root_coverage:
        audit["root_conditioned_coverage"] = _audit_root_coverage(
            root / "manifests", labels_by_id, args.chunk_rows, args.progress_every
        )
    audit["decision"] = (
        "pass"
        if audit["integrity_gates_passed"] and headroom["decision"] == "pass"
        else "fail"
    )
    audit["blocked_stages"] = [] if audit["decision"] == "pass" else ["E6", "E7"]
    _write_json(output, audit)

    print(
        json.dumps(
            {
                "decision": audit["decision"],
                "integrity_gates_passed": audit["integrity_gates_passed"],
                "section_results": {
                    name: section["passed"] for name, section in sorted(sections.items())
                },
                "headroom": {
                    "decision": headroom["decision"],
                    "failed_gates": headroom["failed_gates"],
                    "matches_recorded_gate": headroom["matches_recorded_gate"],
                },
                "root_coverage": {
                    modality: {
                        "cases_covered_both_sides": row["cases_covered_both_sides"],
                        "cases_without_both_sides": row["cases_without_both_sides"],
                        "roots_without_both_sides": row["roots_without_both_sides"],
                    }
                    for modality, row in sorted(
                        audit.get("root_conditioned_coverage", {})
                        .get("modalities", {})
                        .items()
                    )
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
