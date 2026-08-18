#!/usr/bin/env python3
"""Extract full P2 metric feature bundles for GAIA main and RE2-OB."""

import argparse
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
    MetricFeatureAccumulator,
    metric_series_contrasts,
    metric_series_contrasts_many,
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


def _write_json(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _common_services(inputs) -> tuple:
    if not inputs:
        raise ValueError("full extraction requires at least one case")
    services = inputs[0].services
    if any(case_input.services != services for case_input in inputs):
        raise ValueError("full extraction requires one candidate set per dataset")
    return services


def _feature_rows(inputs, accumulator):
    for case_index, case_input in enumerate(inputs):
        for service in case_input.services:
            names, values, observed = accumulator.feature_vector(
                case_index, service
            )
            yield FeatureRow(
                case_id=case_input.case_id,
                service=service,
                extractor=EXTRACTOR,
                feature_names=names,
                values=values,
                observed=observed,
            )


def _extract_gaia(
    inputs,
    manifest_directory: Path,
    min_samples: int,
    score_cap: float,
    progress_every: int,
):
    services = _common_services(inputs)
    accumulator = MetricFeatureAccumulator(len(inputs), services)
    anchors = np.asarray(
        [int(case_input.anchor_time) for case_input in inputs], dtype=np.int64
    )
    source = _read_jsonl(manifest_directory / "sources.jsonl")[0]
    series = _gaia_series(Path(source["metrics_directory"]))
    source_files = 0
    for index, (_, record) in enumerate(series, start=1):
        timestamps, values = _read_gaia_series(record["paths"])
        batches = metric_series_contrasts_many(
            timestamps,
            values,
            anchors,
            onset_seconds=DEFAULT_ONSET_SECONDS,
            min_samples=min_samples,
            score_cap=score_cap,
        )
        for service, _ in record["targets"]:
            accumulator.update(service, batches)
        source_files += len(record["paths"])
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(series)
        ):
            print(
                "[p2-metric] GAIA logical series {}/{}".format(
                    index, len(series)
                ),
                flush=True,
            )
    return accumulator, {
        "processed_logical_series": len(series),
        "processed_source_files": source_files,
    }


def _extract_re2(
    inputs,
    manifest_directory: Path,
    min_samples: int,
    score_cap: float,
    progress_every: int,
):
    services = _common_services(inputs)
    accumulator = MetricFeatureAccumulator(len(inputs), services)
    source_by_case = {
        row["case_id"]: row
        for row in _read_jsonl(manifest_directory / "sources.jsonl")
    }
    processed_columns = 0
    for case_index, case_input in enumerate(inputs):
        source = source_by_case[case_input.case_id]
        frame = pd.read_csv(source["metrics_path"])
        numeric_time = pd.to_numeric(frame["time"], errors="coerce")
        valid_time = numeric_time.notna()
        timestamps = np.floor(
            numeric_time.loc[valid_time].to_numpy(dtype=np.float64) * 1000.0
        ).astype(np.int64)
        anchor_ms = int(float(case_input.anchor_time) * 1000)
        columns_by_service = _service_columns(frame.columns, services)
        for service in services:
            for column in columns_by_service.get(service, ()):
                values = pd.to_numeric(
                    frame.loc[valid_time, column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                accumulator.update_case(
                    service,
                    case_index,
                    metric_series_contrasts(
                        timestamps,
                        values,
                        anchor_ms,
                        onset_seconds=DEFAULT_ONSET_SECONDS,
                        min_samples=min_samples,
                        score_cap=score_cap,
                    ),
                )
                processed_columns += 1
        completed = case_index + 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == len(inputs)
        ):
            print(
                "[p2-metric] RE2-OB cases {}/{}".format(
                    completed, len(inputs)
                ),
                flush=True,
            )
    return accumulator, {
        "processed_candidate_metric_columns": processed_columns,
        "processed_cases": len(inputs),
    }


def _write_dataset(
    output_directory: Path,
    dataset: str,
    inputs,
    accumulator,
    extraction,
    bindings,
    min_samples: int,
    score_cap: float,
):
    manifest = write_feature_bundle(
        str(output_directory),
        dataset,
        inputs,
        _feature_rows(inputs, accumulator),
        extractor_config={
            "duplicate_timestamp_reduce": "mean of finite values",
            "minimum_samples_per_segment": min_samples,
            "onset_candidates_seconds": list(DEFAULT_ONSET_SECONDS),
            "post_seconds": 300,
            "pre_seconds": 300,
            "score_cap": score_cap,
            "smoke": False,
            **extraction,
        },
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
        "index_sha256": manifest["files"]["index.jsonl"]["sha256"],
        "manifest_sha256": _sha256(output_directory / "manifest.json"),
        "observed_sha256": manifest["files"]["observed.npy"]["sha256"],
        "service_row_count": manifest["service_row_count"],
        "values_sha256": manifest["files"]["values.npy"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="artifacts/p1/manifests")
    parser.add_argument("--split-root", default="artifacts/p1/splits")
    parser.add_argument("--inclusion-root", default="artifacts/p1/inclusion/gaia")
    parser.add_argument(
        "--source-snapshot", default="artifacts/p1/source_snapshots/re2ob"
    )
    parser.add_argument(
        "--telemetry-diagnostics",
        default="artifacts/p1/telemetry_diagnostics.json",
    )
    parser.add_argument("--output-root", default="artifacts/p2/features")
    parser.add_argument(
        "--summary", default="artifacts/p2/metric_feature_summary.json"
    )
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--score-cap", type=float, default=20.0)
    parser.add_argument("--gaia-progress-every", type=int, default=25)
    parser.add_argument("--re2-progress-every", type=int, default=10)
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root)
    split_root = Path(args.split_root)
    inclusion_root = Path(args.inclusion_root)
    output_root = Path(args.output_root)

    required_bindings = (
        manifest_root / "gaia" / "manifest.json",
        manifest_root / "re2ob" / "manifest.json",
        inclusion_root / "manifest.json",
        split_root / "gaia" / "assignments.jsonl",
        split_root / "gaia" / "split_manifest.json",
        split_root / "re2ob" / "assignments.jsonl",
        split_root / "re2ob" / "split_manifest.json",
        Path(args.source_snapshot) / "manifest.json",
        Path(args.telemetry_diagnostics),
    )
    missing_bindings = [str(path) for path in required_bindings if not path.is_file()]
    if missing_bindings:
        raise FileNotFoundError(
            "required source bindings are missing: {}".format(missing_bindings)
        )

    gaia_inputs, _ = read_manifest_cases(str(manifest_root / "gaia"))
    main_ids = {
        row["case_id"]
        for row in _read_jsonl(inclusion_root / "main_cohort.jsonl")
    }
    gaia_main = tuple(
        sorted(
            (case_input for case_input in gaia_inputs if case_input.case_id in main_ids),
            key=lambda row: row.case_id,
        )
    )
    if len(gaia_main) != len(main_ids):
        raise ValueError("GAIA main cohort does not match prediction inputs")
    gaia_accumulator, gaia_extraction = _extract_gaia(
        gaia_main,
        manifest_root / "gaia",
        args.min_samples,
        args.score_cap,
        args.gaia_progress_every,
    )
    gaia_summary = _write_dataset(
        output_root / "gaia_main" / EXTRACTOR,
        "GAIA-MicroSS-2021-07/main",
        gaia_main,
        gaia_accumulator,
        gaia_extraction,
        {
            "dataset_manifest_sha256": _sha256(
                manifest_root / "gaia" / "manifest.json"
            ),
            "inclusion_manifest_sha256": _sha256(
                inclusion_root / "manifest.json"
            ),
            "split_assignment_sha256": _sha256(
                split_root / "gaia" / "assignments.jsonl"
            ),
            "split_manifest_sha256": _sha256(
                split_root / "gaia" / "split_manifest.json"
            ),
            "telemetry_diagnostics_sha256": _sha256(
                Path(args.telemetry_diagnostics)
            ),
        },
        args.min_samples,
        args.score_cap,
    )
    del gaia_accumulator

    re2_inputs, _ = read_manifest_cases(str(manifest_root / "re2ob"))
    re2_inputs = tuple(sorted(re2_inputs, key=lambda row: row.case_id))
    re2_accumulator, re2_extraction = _extract_re2(
        re2_inputs,
        manifest_root / "re2ob",
        args.min_samples,
        args.score_cap,
        args.re2_progress_every,
    )
    re2_summary = _write_dataset(
        output_root / "re2ob" / EXTRACTOR,
        "RCAEval-RE2-OB",
        re2_inputs,
        re2_accumulator,
        re2_extraction,
        {
            "dataset_manifest_sha256": _sha256(
                manifest_root / "re2ob" / "manifest.json"
            ),
            "source_snapshot_manifest_sha256": _sha256(
                Path(args.source_snapshot) / "manifest.json"
            ),
            "split_assignment_sha256": _sha256(
                split_root / "re2ob" / "assignments.jsonl"
            ),
            "split_manifest_sha256": _sha256(
                split_root / "re2ob" / "split_manifest.json"
            ),
            "telemetry_diagnostics_sha256": _sha256(
                Path(args.telemetry_diagnostics)
            ),
        },
        args.min_samples,
        args.score_cap,
    )
    summary = {
        "extractor": EXTRACTOR,
        "gaia_main": gaia_summary,
        "re2ob": re2_summary,
    }
    _write_json(Path(args.summary), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
