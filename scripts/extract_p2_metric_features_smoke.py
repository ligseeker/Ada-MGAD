#!/usr/bin/env python3
"""Extract a bounded real-data smoke bundle for P2 metric representations."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p1_metric_change import (
    _gaia_series,
    _read_gaia_series,
    _service_columns,
)
from src.data import read_manifest_cases
from src.features import (
    DEFAULT_ONSET_SECONDS,
    FeatureRow,
    aggregate_metric_contrasts,
    metric_series_contrasts,
    verify_feature_bundle,
    write_feature_bundle,
)


EXTRACTOR = "p2_metric_summary_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle)


def _id_digest(case_ids) -> str:
    payload = "".join("{}\n".format(case_id) for case_id in sorted(case_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_inputs(inputs, limit):
    if limit < 1:
        raise ValueError("smoke case limits must be positive")
    return tuple(sorted(inputs, key=lambda row: row.case_id)[:limit])


def _feature_rows(inputs, contrasts_by_pair, total_series_by_pair):
    rows = []
    for case_input in inputs:
        for service in case_input.services:
            names, values, observed = aggregate_metric_contrasts(
                tuple(contrasts_by_pair[(case_input.case_id, service)]),
                total_series_count=total_series_by_pair.get(
                    (case_input.case_id, service), 0
                ),
            )
            rows.append(
                FeatureRow(
                    case_id=case_input.case_id,
                    service=service,
                    extractor=EXTRACTOR,
                    feature_names=names,
                    values=values,
                    observed=observed,
                )
            )
    return tuple(rows)


def _extract_gaia(inputs, manifest_directory, max_series, min_samples, score_cap):
    if max_series < 1:
        raise ValueError("max_gaia_series must be positive")
    source = _read_jsonl(manifest_directory / "sources.jsonl")[0]
    all_series = _gaia_series(Path(source["metrics_directory"]))
    selected_series = all_series[:max_series]
    contrasts_by_pair = defaultdict(list)
    total_series_by_pair = defaultdict(int)
    for index, (_, record) in enumerate(selected_series, start=1):
        timestamps, values = _read_gaia_series(record["paths"])
        for service, _ in record["targets"]:
            for case_input in inputs:
                total_series_by_pair[(case_input.case_id, service)] += 1
                contrasts_by_pair[(case_input.case_id, service)].append(
                    metric_series_contrasts(
                        timestamps,
                        values,
                        int(case_input.anchor_time),
                        onset_seconds=DEFAULT_ONSET_SECONDS,
                        min_samples=min_samples,
                        score_cap=score_cap,
                    )
                )
        print(
            "[p2-metric-smoke] GAIA series {}/{}".format(
                index, len(selected_series)
            ),
            flush=True,
        )
    rows = _feature_rows(inputs, contrasts_by_pair, total_series_by_pair)
    return rows, {
        "available_logical_series": len(all_series),
        "processed_logical_series": len(selected_series),
        "processed_source_files": sum(
            len(record["paths"]) for _, record in selected_series
        ),
    }


def _extract_re2(inputs, manifest_directory, min_samples, score_cap):
    source_by_case = {
        row["case_id"]: row
        for row in _read_jsonl(manifest_directory / "sources.jsonl")
    }
    contrasts_by_pair = defaultdict(list)
    total_series_by_pair = defaultdict(int)
    processed_columns = 0
    for index, case_input in enumerate(inputs, start=1):
        source = source_by_case[case_input.case_id]
        frame = pd.read_csv(source["metrics_path"])
        numeric_time = pd.to_numeric(frame["time"], errors="coerce")
        valid_time = numeric_time.notna()
        timestamps = np.floor(
            numeric_time.loc[valid_time].to_numpy(dtype=np.float64) * 1000.0
        ).astype(np.int64)
        anchor_ms = int(float(case_input.anchor_time) * 1000)
        columns_by_service = _service_columns(frame.columns, case_input.services)
        for service in case_input.services:
            columns = columns_by_service.get(service, ())
            total_series_by_pair[(case_input.case_id, service)] = len(columns)
            for column in columns:
                values = pd.to_numeric(
                    frame.loc[valid_time, column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                contrasts_by_pair[(case_input.case_id, service)].append(
                    metric_series_contrasts(
                        timestamps,
                        values,
                        anchor_ms,
                        onset_seconds=DEFAULT_ONSET_SECONDS,
                        min_samples=min_samples,
                        score_cap=score_cap,
                    )
                )
                processed_columns += 1
        print(
            "[p2-metric-smoke] RE2 cases {}/{}".format(index, len(inputs)),
            flush=True,
        )
    rows = _feature_rows(inputs, contrasts_by_pair, total_series_by_pair)
    return rows, {
        "processed_candidate_metric_columns": processed_columns,
        "processed_cases": len(inputs),
    }


def _write_dataset(
    output_directory,
    dataset,
    inputs,
    rows,
    extraction,
    bindings,
    min_samples,
    score_cap,
):
    config = {
        "duplicate_timestamp_reduce": "mean of finite values",
        "minimum_samples_per_segment": min_samples,
        "onset_candidates_seconds": list(DEFAULT_ONSET_SECONDS),
        "post_seconds": 300,
        "pre_seconds": 300,
        "score_cap": score_cap,
        "smoke": True,
        **extraction,
    }
    manifest = write_feature_bundle(
        str(output_directory),
        dataset,
        inputs,
        rows,
        extractor_config=config,
        source_bindings={
            **bindings,
            "selected_case_ids_sha256": _id_digest(
                case_input.case_id for case_input in inputs
            ),
        },
    )
    verify_feature_bundle(str(output_directory), inputs)
    return {
        "case_count": manifest["case_count"],
        "feature_count": manifest["feature_count"],
        "manifest_sha256": _sha256(output_directory / "manifest.json"),
        "observed_value_count": sum(sum(row.observed) for row in rows),
        "service_row_count": manifest["service_row_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--inclusion-root", default="artifacts/p1/inclusion/gaia")
    parser.add_argument(
        "--source-snapshot", default="artifacts/p1/source_snapshots/re2ob"
    )
    parser.add_argument("--telemetry-diagnostics", default="artifacts/p1/telemetry_diagnostics.json")
    parser.add_argument("--output-root", default="artifacts/p2/features_smoke")
    parser.add_argument("--max-gaia-cases", type=int, default=3)
    parser.add_argument("--max-gaia-series", type=int, default=5)
    parser.add_argument("--max-re2ob-cases", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--score-cap", type=float, default=20.0)
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    inclusion_root = Path(args.inclusion_root)
    output_root = Path(args.output_root)

    gaia_inputs, _ = read_manifest_cases(str(manifest_root / "gaia"))
    main_ids = {
        row["case_id"] for row in _read_jsonl(inclusion_root / "main_cohort.jsonl")
    }
    gaia_main_inputs = tuple(row for row in gaia_inputs if row.case_id in main_ids)
    selected_gaia = _select_inputs(gaia_main_inputs, args.max_gaia_cases)
    gaia_rows, gaia_extraction = _extract_gaia(
        selected_gaia,
        manifest_root / "gaia",
        args.max_gaia_series,
        args.min_samples,
        args.score_cap,
    )
    gaia_summary = _write_dataset(
        output_root / "gaia_main",
        "GAIA-MicroSS-2021-07/main",
        selected_gaia,
        gaia_rows,
        gaia_extraction,
        {
            "dataset_manifest_sha256": _sha256(manifest_root / "gaia" / "manifest.json"),
            "inclusion_manifest_sha256": _sha256(inclusion_root / "manifest.json"),
            "selected_assignment_sha256": _sha256(split_root / "gaia" / "assignments.jsonl"),
            "telemetry_diagnostics_sha256": _sha256(Path(args.telemetry_diagnostics)),
        },
        args.min_samples,
        args.score_cap,
    )

    re2_inputs, _ = read_manifest_cases(str(manifest_root / "re2ob"))
    selected_re2 = _select_inputs(re2_inputs, args.max_re2ob_cases)
    re2_rows, re2_extraction = _extract_re2(
        selected_re2,
        manifest_root / "re2ob",
        args.min_samples,
        args.score_cap,
    )
    re2_summary = _write_dataset(
        output_root / "re2ob",
        "RCAEval-RE2-OB",
        selected_re2,
        re2_rows,
        re2_extraction,
        {
            "dataset_manifest_sha256": _sha256(manifest_root / "re2ob" / "manifest.json"),
            "selected_assignment_sha256": _sha256(split_root / "re2ob" / "assignments.jsonl"),
            "source_snapshot_manifest_sha256": _sha256(Path(args.source_snapshot) / "manifest.json"),
            "telemetry_diagnostics_sha256": _sha256(Path(args.telemetry_diagnostics)),
        },
        args.min_samples,
        args.score_cap,
    )
    print(json.dumps({"gaia_main": gaia_summary, "re2ob": re2_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
