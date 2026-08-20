#!/usr/bin/env python
"""E3: assign RE2-TT to the frozen 5-fold protocol.

Imports the P1 split machinery rather than reimplementing it, so the folds are
produced by exactly the code that produced RE2-OB's: same
``assign_balanced_group_folds``, same ``SELECTION_THRESHOLDS``, same
``AXIS_WEIGHTS``, same ``SPLIT_SCHEMA_VERSION``, seed 20260819. Writing under
``artifacts/ext/re2tt/splits/`` keeps ``scripts/prepare_p1_splits.py`` — which
rewrites both frozen dataset subtrees in one run — untouched.

RE2-TT case groups are singletons (one official failure-case directory per case),
so the RE2-OB branch applies verbatim: grouped-stratified over root x fault.
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import assign_balanced_group_folds

from scripts.prepare_ext_re2tt_manifests import _assert_isolated
from scripts.prepare_p1_splits import (
    AXIS_WEIGHTS,
    SELECTION_THRESHOLDS,
    SPLIT_SCHEMA_VERSION,
    _assignment_records,
    _load_bundle,
    _source_binding,
    _strata,
    _summarize,
    _verify_split_output,
    _write_json,
    _write_jsonl,
)


DEFAULT_MANIFEST = "artifacts/ext/re2tt/manifests"
DEFAULT_OUTPUT = "artifacts/ext/re2tt/splits"
DEFAULT_DIAGNOSTICS = "artifacts/ext/re2tt/split_diagnostics.json"


def prepare_splits(
    manifest_directory: Path,
    output_directory: Path,
    diagnostics_output: Path,
    n_folds: int,
    seed: int,
):
    manifest, inputs, labels, groups = _load_bundle(manifest_directory)
    assignments = assign_balanced_group_folds(
        groups,
        _strata(labels),
        n_folds=n_folds,
        seed=seed,
        axis_weights=AXIS_WEIGHTS,
    )
    anchors = {case_id: int(row["anchor_time"]) for case_id, row in inputs.items()}
    summary = _summarize(assignments, groups, labels, anchors, n_folds)
    if not summary["selection_eligible"]:
        raise ValueError(
            "RE2-TT stratified folds do not satisfy the frozen selection criteria"
        )

    files = {
        "assignments.jsonl": _write_jsonl(
            output_directory / "assignments.jsonl",
            _assignment_records(assignments, groups),
        )
    }
    split_manifest = {
        "algorithm": "grouped_stratified_5fold",
        "axis_weights": AXIS_WEIGHTS,
        "case_count": len(inputs),
        "dataset": manifest["dataset"],
        "files": files,
        "fold_count": n_folds,
        "seed": seed,
        "selection_reason": "singleton case groups with balanced root/fault strata",
        "source": _source_binding(manifest_directory, manifest),
        "split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    _write_json(output_directory / "split_manifest.json", split_manifest)
    _verify_split_output(output_directory, len(inputs))

    diagnostics = {
        "axis_weights": AXIS_WEIGHTS,
        "extension": "re2tt_protocol_extension",
        "protocol": {
            "fold_count": n_folds,
            "group_integrity": "hard constraint",
            "joint_stratum": "soft balance only; not a per-fold coverage constraint",
            "seed": seed,
            "selection_thresholds": SELECTION_THRESHOLDS,
        },
        "re2tt": {
            "candidate": summary,
            "selected": "grouped_stratified_5fold",
            "source": _source_binding(manifest_directory, manifest),
        },
        "split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    _write_json(diagnostics_output, diagnostics)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-directory", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-directory", default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics-output", default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=20260819, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = Path(args.output_directory)
    diagnostics_output = Path(args.diagnostics_output)
    _assert_isolated(output_directory)
    _assert_isolated(diagnostics_output.parent)

    diagnostics = prepare_splits(
        Path(args.manifest_directory),
        output_directory,
        diagnostics_output,
        n_folds=args.folds,
        seed=args.seed,
    )
    candidate = diagnostics["re2tt"]["candidate"]
    print(
        json.dumps(
            {
                "eligible": candidate["selection_eligible"],
                "fold_sizes": [row["case_count"] for row in candidate["folds"]],
                "max_fault_tv": candidate["max_fault_type_total_variation"],
                "max_root_tv": candidate["max_root_service_total_variation"],
                "output": str(output_directory),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
