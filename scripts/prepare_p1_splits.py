#!/usr/bin/env python
"""Create and diagnose deterministic P1 cross-validation assignments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CaseGroup,
    SplitAssignment,
    assign_balanced_group_folds,
    assign_contiguous_group_folds,
    validate_split_integrity,
    verify_manifest_bundle,
)


SPLIT_SCHEMA_VERSION = "p1_split_manifest_v1"
SELECTION_THRESHOLDS = {
    "max_case_count_relative_deviation": 0.15,
    "max_fault_type_total_variation": 0.10,
    "max_root_service_total_variation": 0.10,
    "require_every_feasible_fault_type_in_every_fold": True,
    "require_every_feasible_root_service_in_every_fold": True,
}
AXIS_WEIGHTS = {"fault_type": 1.0, "joint": 0.25, "root_service": 1.0}


def _read_jsonl(path: Path) -> Tuple[Mapping[str, object], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{} contains invalid JSON".format(path, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise ValueError("{}:{} must be a JSON object".format(path, line_number))
            rows.append(row)
    return tuple(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> Mapping[str, object]:
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


def _write_json(path: Path, record: Mapping[str, object]) -> None:
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


def _load_bundle(directory: Path):
    manifest = verify_manifest_bundle(str(directory))
    input_rows = _read_jsonl(directory / "inputs.jsonl")
    label_rows = _read_jsonl(directory / "labels.jsonl")
    group_rows = _read_jsonl(directory / "groups.jsonl")
    inputs = {str(row["case_id"]): row for row in input_rows}
    labels = {str(row["case_id"]): row for row in label_rows}
    groups = tuple(
        CaseGroup(
            case_id=str(row["case_id"]),
            group_id=str(row["group_id"]),
            group_size=int(row["group_size"]),
            overlap_degree=int(row["overlap_degree"]),
        )
        for row in group_rows
    )
    expected = {row.case_id for row in groups}
    if expected != set(inputs) or expected != set(labels):
        raise ValueError("input/label/group case IDs do not match in {}".format(directory))
    validate_split_integrity(
        tuple(SplitAssignment(case_id, "all") for case_id in sorted(expected)),
        groups,
    )
    return manifest, inputs, labels, groups


def _strata(labels: Mapping[str, Mapping[str, object]]):
    return {
        case_id: {
            "fault_type": str(row["fault_type"]),
            "joint": "{}\x1f{}".format(row["root_service"], row["fault_type"]),
            "root_service": str(row["root_service"]),
        }
        for case_id, row in labels.items()
    }


def _total_variation(
    observed: Mapping[str, int], total: int, reference: Mapping[str, int], reference_total: int
) -> float:
    return 0.5 * sum(
        abs(observed.get(category, 0) / float(total) - count / float(reference_total))
        for category, count in reference.items()
    )


def _summarize(
    assignments: Sequence[SplitAssignment],
    groups: Sequence[CaseGroup],
    labels: Mapping[str, Mapping[str, object]],
    anchors: Mapping[str, int],
    n_folds: int,
) -> Mapping[str, object]:
    validate_split_integrity(assignments, groups)
    split_by_case = {row.case_id: row.split for row in assignments}
    expected_splits = {"fold-{}".format(index) for index in range(n_folds)}
    if set(split_by_case.values()) != expected_splits:
        raise ValueError("assignments do not contain exactly the requested folds")

    group_by_case = {row.case_id: row.group_id for row in groups}
    global_counts = {
        "fault_type": Counter(str(row["fault_type"]) for row in labels.values()),
        "root_service": Counter(str(row["root_service"]) for row in labels.values()),
        "joint": Counter(
            "{}\x1f{}".format(row["root_service"], row["fault_type"])
            for row in labels.values()
        ),
    }
    supporting_groups = {
        axis: defaultdict(set) for axis in ("fault_type", "root_service", "joint")
    }
    for case_id, row in labels.items():
        supporting_groups["fault_type"][str(row["fault_type"])].add(
            group_by_case[case_id]
        )
        supporting_groups["root_service"][str(row["root_service"])].add(
            group_by_case[case_id]
        )
        supporting_groups["joint"][
            "{}\x1f{}".format(row["root_service"], row["fault_type"])
        ].add(group_by_case[case_id])
    feasible_categories = {
        axis: sorted(
            category
            for category, group_ids in categories.items()
            if len(group_ids) >= n_folds
        )
        for axis, categories in supporting_groups.items()
    }
    infeasible_categories = {
        axis: sorted(set(global_counts[axis]) - set(feasible_categories[axis]))
        for axis in global_counts
    }

    cases_by_fold = defaultdict(list)
    for case_id, split in split_by_case.items():
        cases_by_fold[split].append(case_id)
    fold_rows = []
    target_size = len(assignments) / float(n_folds)
    max_size_deviation = 0.0
    maximum_tv = {axis: 0.0 for axis in global_counts}
    coverage_failures = {axis: [] for axis in global_counts}
    for fold_index in range(n_folds):
        split = "fold-{}".format(fold_index)
        case_ids = sorted(cases_by_fold[split])
        counts = {
            "fault_type": Counter(
                str(labels[case_id]["fault_type"]) for case_id in case_ids
            ),
            "root_service": Counter(
                str(labels[case_id]["root_service"]) for case_id in case_ids
            ),
            "joint": Counter(
                "{}\x1f{}".format(
                    labels[case_id]["root_service"], labels[case_id]["fault_type"]
                )
                for case_id in case_ids
            ),
        }
        televisions = {
            axis: _total_variation(
                counts[axis], len(case_ids), global_counts[axis], len(assignments)
            )
            for axis in global_counts
        }
        missing = {
            axis: sorted(set(feasible_categories[axis]) - set(counts[axis]))
            for axis in global_counts
        }
        for axis in global_counts:
            maximum_tv[axis] = max(maximum_tv[axis], televisions[axis])
            if missing[axis]:
                coverage_failures[axis].append(
                    {"fold": split, "missing_categories": missing[axis]}
                )
        size_deviation = abs(len(case_ids) - target_size) / target_size
        max_size_deviation = max(max_size_deviation, size_deviation)
        fold_rows.append(
            {
                "anchor_max": max(anchors[case_id] for case_id in case_ids),
                "anchor_min": min(anchors[case_id] for case_id in case_ids),
                "case_count": len(case_ids),
                "case_count_relative_deviation": size_deviation,
                "fault_type_counts": dict(sorted(counts["fault_type"].items())),
                "fault_type_total_variation": televisions["fault_type"],
                "fold": split,
                "joint_category_count": len(counts["joint"]),
                "joint_total_variation": televisions["joint"],
                "root_service_counts": dict(sorted(counts["root_service"].items())),
                "root_service_total_variation": televisions["root_service"],
            }
        )

    eligible = (
        max_size_deviation
        <= SELECTION_THRESHOLDS["max_case_count_relative_deviation"]
        and maximum_tv["fault_type"]
        <= SELECTION_THRESHOLDS["max_fault_type_total_variation"]
        and maximum_tv["root_service"]
        <= SELECTION_THRESHOLDS["max_root_service_total_variation"]
        and not coverage_failures["fault_type"]
        and not coverage_failures["root_service"]
    )
    return {
        "case_count": len(assignments),
        "coverage_failures": coverage_failures,
        "feasible_categories": feasible_categories,
        "fold_count": n_folds,
        "folds": fold_rows,
        "infeasible_categories": infeasible_categories,
        "integrity": "pass",
        "max_case_count_relative_deviation": max_size_deviation,
        "max_fault_type_total_variation": maximum_tv["fault_type"],
        "max_joint_total_variation": maximum_tv["joint"],
        "max_root_service_total_variation": maximum_tv["root_service"],
        "selection_eligible": eligible,
    }


def _assignment_records(
    assignments: Sequence[SplitAssignment], groups: Sequence[CaseGroup]
) -> Tuple[Mapping[str, object], ...]:
    group_by_case = {row.case_id: row.group_id for row in groups}
    return tuple(
        {
            "case_id": row.case_id,
            "fold": int(row.split.split("-", 1)[1]),
            "group_id": group_by_case[row.case_id],
            "split": row.split,
        }
        for row in sorted(assignments, key=lambda value: value.case_id)
    )


def _source_binding(
    directory: Path, manifest: Mapping[str, object]
) -> Mapping[str, object]:
    return {
        "directory": str(directory),
        "groups_sha256": manifest["files"]["groups.jsonl"]["sha256"],
        "inputs_sha256": manifest["files"]["inputs.jsonl"]["sha256"],
        "labels_sha256": manifest["files"]["labels.jsonl"]["sha256"],
        "manifest_sha256": _sha256(directory / "manifest.json"),
    }


def _verify_split_output(directory: Path, expected_case_count: int) -> None:
    split_manifest = json.loads(
        (directory / "split_manifest.json").read_text(encoding="utf-8")
    )
    if split_manifest.get("split_schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema")
    if split_manifest.get("case_count") != expected_case_count:
        raise ValueError("split manifest case_count mismatch")
    for filename, expected in split_manifest.get("files", {}).items():
        path = directory / filename
        rows = _read_jsonl(path)
        if len(rows) != expected.get("rows") or _sha256(path) != expected.get("sha256"):
            raise ValueError("{} row count or checksum mismatch".format(path))
        case_ids = [str(row.get("case_id", "")) for row in rows]
        if len(rows) != expected_case_count or len(set(case_ids)) != len(case_ids):
            raise ValueError("{} does not cover each case exactly once".format(path))


def prepare_splits(
    manifest_root: Path,
    output_root: Path,
    diagnostics_output: Path,
    n_folds: int,
    seed: int,
) -> Mapping[str, object]:
    gaia_dir = manifest_root / "gaia"
    re2ob_dir = manifest_root / "re2ob"
    gaia_manifest, gaia_inputs, gaia_labels, gaia_groups = _load_bundle(gaia_dir)
    re2_manifest, re2_inputs, re2_labels, re2_groups = _load_bundle(re2ob_dir)

    gaia_grouped = assign_balanced_group_folds(
        gaia_groups,
        _strata(gaia_labels),
        n_folds=n_folds,
        seed=seed,
        axis_weights=AXIS_WEIGHTS,
    )
    gaia_temporal = assign_contiguous_group_folds(
        gaia_groups,
        {case_id: int(row["anchor_time"]) for case_id, row in gaia_inputs.items()},
        n_folds=n_folds,
    )
    re2_stratified = assign_balanced_group_folds(
        re2_groups,
        _strata(re2_labels),
        n_folds=n_folds,
        seed=seed,
        axis_weights=AXIS_WEIGHTS,
    )

    gaia_anchors = {
        case_id: int(row["anchor_time"]) for case_id, row in gaia_inputs.items()
    }
    re2_anchors = {
        case_id: int(row["anchor_time"]) for case_id, row in re2_inputs.items()
    }
    gaia_candidates = {
        "grouped_stratified_5fold": _summarize(
            gaia_grouped, gaia_groups, gaia_labels, gaia_anchors, n_folds
        ),
        "temporal_block_5fold": _summarize(
            gaia_temporal, gaia_groups, gaia_labels, gaia_anchors, n_folds
        ),
    }
    re2_summary = _summarize(
        re2_stratified, re2_groups, re2_labels, re2_anchors, n_folds
    )
    if gaia_candidates["temporal_block_5fold"]["selection_eligible"]:
        gaia_selected_name = "temporal_block_5fold"
        gaia_selected = gaia_temporal
        gaia_reason = (
            "temporal block met all predeclared balance and coverage thresholds; "
            "selected for stronger temporal separation"
        )
    elif gaia_candidates["grouped_stratified_5fold"]["selection_eligible"]:
        gaia_selected_name = "grouped_stratified_5fold"
        gaia_selected = gaia_grouped
        gaia_reason = (
            "temporal block failed at least one predeclared balance or coverage "
            "threshold; selected the eligible context-group-preserving alternative"
        )
    else:
        raise ValueError("neither GAIA split candidate satisfies the selection criteria")
    if not re2_summary["selection_eligible"]:
        raise ValueError("RE2-OB stratified folds do not satisfy the selection criteria")

    gaia_output = output_root / "gaia"
    re2_output = output_root / "re2ob"
    gaia_files = {
        "grouped_stratified_5fold.jsonl": _write_jsonl(
            gaia_output / "grouped_stratified_5fold.jsonl",
            _assignment_records(gaia_grouped, gaia_groups),
        ),
        "temporal_block_5fold.jsonl": _write_jsonl(
            gaia_output / "temporal_block_5fold.jsonl",
            _assignment_records(gaia_temporal, gaia_groups),
        ),
        "assignments.jsonl": _write_jsonl(
            gaia_output / "assignments.jsonl",
            _assignment_records(gaia_selected, gaia_groups),
        ),
    }
    re2_files = {
        "assignments.jsonl": _write_jsonl(
            re2_output / "assignments.jsonl",
            _assignment_records(re2_stratified, re2_groups),
        )
    }

    gaia_split_manifest = {
        "algorithm": gaia_selected_name,
        "axis_weights": AXIS_WEIGHTS,
        "case_count": len(gaia_inputs),
        "dataset": gaia_manifest["dataset"],
        "files": gaia_files,
        "fold_count": n_folds,
        "seed": seed,
        "selection_reason": gaia_reason,
        "source": _source_binding(gaia_dir, gaia_manifest),
        "split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    re2_split_manifest = {
        "algorithm": "grouped_stratified_5fold",
        "axis_weights": AXIS_WEIGHTS,
        "case_count": len(re2_inputs),
        "dataset": re2_manifest["dataset"],
        "files": re2_files,
        "fold_count": n_folds,
        "seed": seed,
        "selection_reason": "singleton case groups with balanced root/fault strata",
        "source": _source_binding(re2ob_dir, re2_manifest),
        "split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    _write_json(gaia_output / "split_manifest.json", gaia_split_manifest)
    _write_json(re2_output / "split_manifest.json", re2_split_manifest)
    _verify_split_output(gaia_output, len(gaia_inputs))
    _verify_split_output(re2_output, len(re2_inputs))

    diagnostics = {
        "axis_weights": AXIS_WEIGHTS,
        "gaia": {
            "candidates": gaia_candidates,
            "selected": gaia_selected_name,
            "selection_reason": gaia_reason,
            "source": _source_binding(gaia_dir, gaia_manifest),
        },
        "protocol": {
            "fold_count": n_folds,
            "gaia_context_seconds": 300,
            "selection_thresholds": SELECTION_THRESHOLDS,
            "group_integrity": "hard constraint",
            "joint_stratum": "soft balance only; not a per-fold coverage constraint",
            "seed": seed,
        },
        "re2ob": {
            "candidate": re2_summary,
            "selected": "grouped_stratified_5fold",
            "source": _source_binding(re2ob_dir, re2_manifest),
        },
        "split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    _write_json(diagnostics_output, diagnostics)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--diagnostics-output", required=True)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=20260819, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = prepare_splits(
        Path(args.manifest_root),
        Path(args.output_root),
        Path(args.diagnostics_output),
        n_folds=args.folds,
        seed=args.seed,
    )
    summary = {
        "gaia": {
            name: {
                "eligible": values["selection_eligible"],
                "fold_sizes": [row["case_count"] for row in values["folds"]],
                "max_fault_tv": values["max_fault_type_total_variation"],
                "max_root_tv": values["max_root_service_total_variation"],
            }
            for name, values in diagnostics["gaia"]["candidates"].items()
        },
        "gaia_selected": diagnostics["gaia"]["selected"],
        "re2ob": {
            "eligible": diagnostics["re2ob"]["candidate"]["selection_eligible"],
            "fold_sizes": [
                row["case_count"]
                for row in diagnostics["re2ob"]["candidate"]["folds"]
            ],
            "max_fault_tv": diagnostics["re2ob"]["candidate"][
                "max_fault_type_total_variation"
            ],
            "max_root_tv": diagnostics["re2ob"]["candidate"][
                "max_root_service_total_variation"
            ],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
