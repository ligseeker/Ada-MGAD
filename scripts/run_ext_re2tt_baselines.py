#!/usr/bin/env python
"""E4: RE2-TT sanity baselines (B0/B1) + metric-change baseline (B2) + headroom gate.

Every baseline is produced by the P1 implementations, imported verbatim:
``run_p1_sanity_baselines._run_dataset`` for B0/B1 and
``run_p1_metric_change._extract_re2ob`` for B2, with the frozen metric-change
config (300 s half-open window, 2 samples per side, top-5 features, cap 20.0) and
seed 20260819. The P1 entry points themselves rewrite both frozen dataset subtrees
in one run, so they are not invoked; only this driver is, and it writes exclusively
under ``artifacts/ext/re2tt/``.

The headroom gate is the pre-registered E4 criterion from
``docs/RE2TT_EXTENSION_PROTOCOL.md`` section 6.3. Its purpose is to decide whether
RE2-TT has room for a Track C effect to be visible at all: RE2-OB failed H-1 with a
B2 root-macro Avg@5 of 0.933333, which is why its P2-G4 reading could not separate
"H1 ineffective" from "dataset saturated". A gate failure here is a finding about
the benchmark, not about H1, and it blocks E6/E7 rather than changing any threshold.
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_ranking_report

from scripts.prepare_ext_re2tt_manifests import _assert_isolated
from scripts.run_p1_metric_change import (
    _compact_result,
    _coverage_audit,
    _coverage_slice_reports,
    _extract_re2ob,
)
from scripts.run_p1_sanity_baselines import (
    BASELINE_SCHEMA_VERSION,
    _load_split,
    _run_dataset,
    _write_baseline_result,
    _write_json,
)


DEFAULT_MANIFEST = "artifacts/ext/re2tt/manifests"
DEFAULT_SPLIT = "artifacts/ext/re2tt/splits"
DEFAULT_OUTPUT = "artifacts/ext/re2tt/baselines"
DEFAULT_SUMMARY = "artifacts/ext/re2tt/baseline_summary.json"
DEFAULT_GATE = "artifacts/ext/re2tt/headroom_gate.json"
FROZEN_P1_SUMMARY = "artifacts/p1/baseline_summary.json"

GATE_SCHEMA_VERSION = "ext_re2tt_headroom_gate_v1"
PRIMARY_ENDPOINT = "Avg@5"
KEY_SECONDARY_ENDPOINT = "AC@1"
SATURATION_CEILING = 0.90


def _metric_change_config(args) -> dict:
    """Rebuild run_p1_metric_change.run_metric_change's config dict verbatim."""

    return {
        "duplicate_timestamp_reduce": "mean of finite values",
        "feature_aggregation": "mean of top-k standardized feature shifts",
        "minimum_samples_per_side": args.min_samples_per_side,
        "missing_service_fallback": (
            "observed services first; missing services alphabetical last"
        ),
        "score_cap": args.score_cap,
        "score_scale": (
            "pooled within-window population standard deviation + relative epsilon"
        ),
        "top_k_features": args.top_k_features,
        "window_semantics": "half-open [t0-window,t0) vs [t0,t0+window)",
        "window_seconds": args.window_seconds,
    }


def _run_metric_change(
    manifest_directory: Path,
    split_directory: Path,
    output_directory: Path,
    args,
):
    config = _metric_change_config(args)
    (
        dataset_manifest,
        split_manifest,
        inputs,
        labels,
        assignments,
    ) = _load_split(manifest_directory, split_directory)
    rankings, details, predictions, extraction_audit = _extract_re2ob(
        inputs,
        manifest_directory,
        args.window_seconds * 1000,
        args.min_samples_per_side,
        args.score_cap,
        args.top_k_features,
        args.max_cases,
        args.progress_every,
        progress_label="RE2-TT cases",
    )
    metrics = dict(evaluate_ranking_report(inputs, labels, rankings))
    metrics["coverage_slices"] = _coverage_slice_reports(
        inputs, labels, rankings, predictions
    )
    training_audit = {
        "coverage": _coverage_audit(predictions, inputs[0].services),
        "extraction": extraction_audit,
        "fit_scope": "none; every score uses only its own case window",
    }
    result = _write_baseline_result(
        output_directory / "metric_change",
        "metric_change",
        config,
        rankings,
        metrics,
        training_audit,
        {row.case_id: row.split for row in assignments},
        dataset_manifest,
        manifest_directory,
        split_manifest,
        split_directory,
        prediction_details=details,
    )
    return config, _compact_result(metrics, result["run_manifest_sha256"])


def _reference_rows(frozen_summary: Path):
    """RE2-OB / GAIA-main comparison values, read from the frozen P1 summary."""

    if not frozen_summary.is_file():
        return {"status": "absent", "path": str(frozen_summary)}
    summary = json.loads(frozen_summary.read_text(encoding="utf-8"))
    rows = {}
    for dataset in ("gaia_main", "re2ob"):
        section = summary["datasets"].get(dataset)
        if not section:
            continue
        rows[dataset] = {
            baseline: section[baseline]["root_service_macro"]
            for baseline in sorted(section)
        }
    return {"status": "read", "path": str(frozen_summary), "root_service_macro": rows}


def _headroom_gate(summary, reference):
    b1 = summary["root_frequency"]["root_service_macro"]
    b2 = summary["metric_change"]["root_service_macro"]
    reference_b2 = (
        reference.get("root_service_macro", {}).get("re2ob", {}).get("metric_change", {})
    )
    checks = [
        {
            "id": "H-1",
            "criterion": "B2 metric_change root-macro {} <= {}".format(
                PRIMARY_ENDPOINT, SATURATION_CEILING
            ),
            "kind": "gate",
            "observed": b2[PRIMARY_ENDPOINT],
            "passed": b2[PRIMARY_ENDPOINT] <= SATURATION_CEILING,
            "re2ob_reference": reference_b2.get(PRIMARY_ENDPOINT),
        },
        {
            "id": "H-2",
            "criterion": "B2 metric_change root-macro {} <= {}".format(
                KEY_SECONDARY_ENDPOINT, SATURATION_CEILING
            ),
            "kind": "gate",
            "observed": b2[KEY_SECONDARY_ENDPOINT],
            "passed": b2[KEY_SECONDARY_ENDPOINT] <= SATURATION_CEILING,
            "re2ob_reference": reference_b2.get(KEY_SECONDARY_ENDPOINT),
        },
        {
            "id": "H-3",
            "criterion": (
                "B2 strictly better than B1 on both the primary and the key "
                "secondary endpoint"
            ),
            "kind": "gate",
            "observed": {
                KEY_SECONDARY_ENDPOINT: b2[KEY_SECONDARY_ENDPOINT]
                - b1[KEY_SECONDARY_ENDPOINT],
                PRIMARY_ENDPOINT: b2[PRIMARY_ENDPOINT] - b1[PRIMARY_ENDPOINT],
            },
            "passed": (
                b2[PRIMARY_ENDPOINT] > b1[PRIMARY_ENDPOINT]
                and b2[KEY_SECONDARY_ENDPOINT] > b1[KEY_SECONDARY_ENDPOINT]
            ),
        },
        {
            "id": "H-4",
            "criterion": "B1 root_frequency root-macro AC@5 (prior degeneracy witness)",
            "kind": "report_only",
            "observed": b1["AC@5"],
            "passed": None,
            "re2ob_reference": (
                reference.get("root_service_macro", {})
                .get("re2ob", {})
                .get("root_frequency", {})
                .get("AC@5")
            ),
        },
    ]
    gates = [check for check in checks if check["kind"] == "gate"]
    return {
        "checks": checks,
        "decision": "pass" if all(check["passed"] for check in gates) else "fail",
        "failed_gates": sorted(
            check["id"] for check in gates if not check["passed"]
        ),
        "gate_count": len(gates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-directory", default=DEFAULT_MANIFEST)
    parser.add_argument("--split-directory", default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--gate-output", default=DEFAULT_GATE)
    parser.add_argument("--frozen-p1-summary", default=FROZEN_P1_SUMMARY)
    parser.add_argument("--random-seed", default=20260819, type=int)
    parser.add_argument("--window-seconds", default=300, type=int)
    parser.add_argument("--min-samples-per-side", default=2, type=int)
    parser.add_argument("--top-k-features", default=5, type=int)
    parser.add_argument("--score-cap", default=20.0, type=float)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--progress-every", default=10, type=int)
    parser.add_argument(
        "--skip-metric-change",
        action="store_true",
        help="produce B0/B1 only; B2 rescans every candidate metric column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_directory = Path(args.manifest_directory)
    split_directory = Path(args.split_directory)
    output_root = Path(args.output_root)
    summary_output = Path(args.summary_output)
    gate_output = Path(args.gate_output)
    for target in (output_root, summary_output.parent, gate_output.parent):
        _assert_isolated(target)

    sanity = _run_dataset(
        manifest_directory, split_directory, output_root, args.random_seed
    )
    summary = {
        baseline: _compact_result(
            result["metrics"], result["run_manifest_sha256"]
        )
        for baseline, result in sanity.items()
    }

    config = None
    if not args.skip_metric_change:
        config, summary["metric_change"] = _run_metric_change(
            manifest_directory, split_directory, output_root, args
        )

    reference = _reference_rows(Path(args.frozen_p1_summary))
    record = {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "datasets": {"re2tt": summary},
        "extension": "re2tt_protocol_extension",
        "random_seed": args.random_seed,
    }
    if config is not None:
        record["metric_change_config"] = config
    _write_json(summary_output, record)

    if "metric_change" in summary:
        gate = {
            "extension": "re2tt_protocol_extension",
            "gate": _headroom_gate(summary, reference),
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "interpretation": (
                "a failed gate is a statement about RE2-TT's usable headroom, not "
                "about H1; it blocks E6/E7 and never relaxes a frozen threshold"
            ),
            "reference": reference,
            "saturation_ceiling": SATURATION_CEILING,
            "source": {
                "baseline_summary": str(summary_output),
                "manifest_directory": str(manifest_directory),
                "split_directory": str(split_directory),
            },
        }
        _write_json(gate_output, gate)
        print(json.dumps(gate["gate"], ensure_ascii=False, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                baseline: {
                    key: round(values["root_service_macro"][key], 6)
                    for key in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")
                }
                for baseline, values in sorted(summary.items())
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
