#!/usr/bin/env python
"""Run the within-case Metric Change Score baseline on GAIA and RE2-OB."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p1_sanity_baselines import (
    BASELINE_SCHEMA_VERSION,
    _load_split,
    _read_jsonl,
    _write_baseline_result,
    _write_json,
)
from src.baselines import metric_change_prediction, window_mean_shift_scores
from src.evaluation import evaluate_ranking_report
from util.GAIA.pre_GAIA import _parse_metric_filename, _target_services_for_metric


class _TopFeatureAccumulator:
    def __init__(self, case_inputs, top_k):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.case_inputs = tuple(case_inputs)
        self.services = self.case_inputs[0].services
        if any(case.services != self.services for case in self.case_inputs):
            raise ValueError("metric-change runner requires one candidate set per dataset")
        self.service_index = {
            service: index for index, service in enumerate(self.services)
        }
        self.top_scores = np.full(
            (len(self.case_inputs), len(self.services), top_k),
            -np.inf,
            dtype=np.float64,
        )
        self.feature_counts = np.zeros(
            (len(self.case_inputs), len(self.services)), dtype=np.int32
        )

    def update_scores(self, service, scores):
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (len(self.case_inputs),):
            raise ValueError("metric scores do not align with case inputs")
        service_index = self.service_index[service]
        valid = np.isfinite(values)
        self.feature_counts[valid, service_index] += 1
        slots = self.top_scores[:, service_index, :]
        minimum_slot = np.argmin(slots, axis=1)
        row_indices = np.arange(len(self.case_inputs))
        replace = valid & (values > slots[row_indices, minimum_slot])
        slots[row_indices[replace], minimum_slot[replace]] = values[replace]

    def update_case_score(self, case_index, service, score):
        if not np.isfinite(score):
            return
        service_index = self.service_index[service]
        self.feature_counts[case_index, service_index] += 1
        slots = self.top_scores[case_index, service_index]
        minimum_slot = int(np.argmin(slots))
        if score > slots[minimum_slot]:
            slots[minimum_slot] = score

    def finalize(self):
        rankings = {}
        details = {}
        predictions = {}
        for case_index, case_input in enumerate(self.case_inputs):
            service_scores = {}
            feature_counts = {}
            for service_index, service in enumerate(self.services):
                finite = self.top_scores[case_index, service_index]
                finite = finite[np.isfinite(finite)]
                feature_counts[service] = int(
                    self.feature_counts[case_index, service_index]
                )
                service_scores[service] = (
                    float(finite.mean()) if finite.size else float("nan")
                )
            prediction = metric_change_prediction(
                case_input, service_scores, feature_counts
            )
            predictions[case_input.case_id] = prediction
            rankings[case_input.case_id] = prediction.ranking
            details[case_input.case_id] = {
                "all_services_fallback": len(prediction.fallback_services)
                == len(case_input.services),
                "fallback_services": list(prediction.fallback_services),
                "observed_feature_counts": dict(
                    sorted(prediction.observed_feature_counts.items())
                ),
                "service_scores": {
                    service: (
                        score if math.isfinite(score) else None
                    )
                    for service, score in sorted(prediction.service_scores.items())
                },
            }
        return rankings, details, predictions


def _coverage_audit(predictions, services):
    fallback_counts = {
        service: sum(
            service in prediction.fallback_services
            for prediction in predictions.values()
        )
        for service in services
    }
    observed_counts = [
        count
        for prediction in predictions.values()
        for count in prediction.observed_feature_counts.values()
    ]
    return {
        "cases_all_services_fallback": sum(
            len(prediction.fallback_services) == len(services)
            for prediction in predictions.values()
        ),
        "cases_any_service_fallback": sum(
            bool(prediction.fallback_services)
            for prediction in predictions.values()
        ),
        "fallback_case_count_by_service": fallback_counts,
        "observed_feature_count": {
            "max": max(observed_counts, default=0),
            "mean": sum(observed_counts) / len(observed_counts)
            if observed_counts
            else 0.0,
            "min": min(observed_counts, default=0),
        },
    }


def _gaia_series(metric_directory: Path):
    grouped = {}
    for path in sorted(metric_directory.glob("*.csv")):
        info = _parse_metric_filename(path.name)
        targets = _target_services_for_metric(info) if info is not None else []
        if not targets:
            continue
        key = str(info["full_name"])
        if key not in grouped:
            grouped[key] = {"paths": [], "targets": tuple(targets)}
        elif grouped[key]["targets"] != tuple(targets):
            raise ValueError("GAIA metric target mapping changed across shards")
        grouped[key]["paths"].append(path)
    return tuple((key, grouped[key]) for key in sorted(grouped))


def _read_gaia_series(paths: Sequence[Path]):
    timestamps = []
    values = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["timestamp", "value"])
        raw_timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        raw_values = pd.to_numeric(frame["value"], errors="coerce")
        valid = raw_timestamps.notna() & raw_values.notna()
        timestamps.append(
            np.floor(raw_timestamps.loc[valid].to_numpy(dtype=np.float64)).astype(
                np.int64
            )
        )
        values.append(raw_values.loc[valid].to_numpy(dtype=np.float64))
    if not timestamps:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.concatenate(timestamps), np.concatenate(values)


def _extract_gaia(
    inputs,
    manifest_directory: Path,
    radius_ms,
    min_samples,
    score_cap,
    top_k,
    max_series,
    progress_every,
):
    source = _read_jsonl(manifest_directory / "sources.jsonl")[0]
    metric_directory = Path(source["metrics_directory"])
    series = _gaia_series(metric_directory)
    selected = series[:max_series] if max_series is not None else series
    accumulator = _TopFeatureAccumulator(inputs, top_k)
    anchors = np.asarray([int(case.anchor_time) for case in inputs], dtype=np.int64)
    source_files = 0
    for index, (_, record) in enumerate(selected, start=1):
        timestamps, values = _read_gaia_series(record["paths"])
        scores = window_mean_shift_scores(
            timestamps,
            values,
            anchors,
            radius_ms,
            min_samples_per_side=min_samples,
            score_cap=score_cap,
        )
        for service, _ in record["targets"]:
            accumulator.update_scores(service, scores)
        source_files += len(record["paths"])
        if progress_every > 0 and (index % progress_every == 0 or index == len(selected)):
            print(
                "[metric-change] GAIA series: {}/{}".format(index, len(selected)),
                file=sys.stderr,
                flush=True,
            )
    rankings, details, predictions = accumulator.finalize()
    return rankings, details, predictions, {
        "available_logical_series": len(series),
        "processed_logical_series": len(selected),
        "processed_source_files": source_files,
        "scan_limited": max_series is not None,
    }


def _service_columns(columns, services):
    result = defaultdict(list)
    for column in columns:
        if column == "time":
            continue
        for service in services:
            if column.startswith(service + "_"):
                result[service].append(column)
                break
    return result


def _extract_re2ob(
    inputs,
    manifest_directory: Path,
    radius_ms,
    min_samples,
    score_cap,
    top_k,
    max_cases,
    progress_every,
):
    source_by_case = {
        str(row["case_id"]): row
        for row in _read_jsonl(manifest_directory / "sources.jsonl")
    }
    accumulator = _TopFeatureAccumulator(inputs, top_k)
    limit = len(inputs) if max_cases is None else min(max_cases, len(inputs))
    metric_columns = 0
    for case_index, case_input in enumerate(inputs[:limit]):
        path = Path(source_by_case[case_input.case_id]["metrics_path"])
        frame = pd.read_csv(path)
        raw_time = pd.to_numeric(frame["time"], errors="coerce")
        valid_time = raw_time.notna()
        timestamps = np.floor(
            raw_time.loc[valid_time].to_numpy(dtype=np.float64) * 1000.0
        ).astype(np.int64)
        anchor = np.asarray([int(float(case_input.anchor_time) * 1000)], dtype=np.int64)
        columns_by_service = _service_columns(frame.columns, case_input.services)
        for service, columns in columns_by_service.items():
            for column in columns:
                values = pd.to_numeric(
                    frame.loc[valid_time, column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                score = window_mean_shift_scores(
                    timestamps,
                    values,
                    anchor,
                    radius_ms,
                    min_samples_per_side=min_samples,
                    score_cap=score_cap,
                )[0]
                accumulator.update_case_score(case_index, service, score)
                metric_columns += 1
        completed = case_index + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == limit):
            print(
                "[metric-change] RE2-OB cases: {}/{}".format(completed, limit),
                file=sys.stderr,
                flush=True,
            )
    rankings, details, predictions = accumulator.finalize()
    return rankings, details, predictions, {
        "available_cases": len(inputs),
        "processed_cases": limit,
        "processed_candidate_metric_columns": metric_columns,
        "scan_limited": max_cases is not None,
    }


def _compact_result(metrics, run_manifest_sha256):
    return {
        "fault_type_macro": metrics["fault_type"]["macro"],
        "overall": metrics["overall"],
        "root_service_macro": metrics["root_service"]["macro"],
        "run_manifest_sha256": run_manifest_sha256,
    }


def _coverage_slice_reports(inputs, labels, rankings, predictions):
    inputs_by_id = {row.case_id: row for row in inputs}
    labels_by_id = {row.case_id: row for row in labels}
    observed_ids = sorted(
        case_id
        for case_id, prediction in predictions.items()
        if len(prediction.fallback_services) < len(inputs_by_id[case_id].services)
    )
    fallback_ids = sorted(set(inputs_by_id) - set(observed_ids))

    def report(case_ids):
        if not case_ids:
            return {"case_count": 0, "status": "empty"}
        return {
            "case_count": len(case_ids),
            "report": evaluate_ranking_report(
                tuple(inputs_by_id[case_id] for case_id in case_ids),
                tuple(labels_by_id[case_id] for case_id in case_ids),
                {case_id: rankings[case_id] for case_id in case_ids},
            ),
            "status": "evaluated",
        }

    return {
        "all_services_fallback": report(fallback_ids),
        "metric_observed": report(observed_ids),
    }


def run_metric_change(args):
    config = {
        "duplicate_timestamp_reduce": "mean of finite values",
        "feature_aggregation": "mean of top-k standardized feature shifts",
        "minimum_samples_per_side": args.min_samples_per_side,
        "missing_service_fallback": "observed services first; missing services alphabetical last",
        "score_cap": args.score_cap,
        "score_scale": "pooled within-window population standard deviation + relative epsilon",
        "top_k_features": args.top_k_features,
        "window_semantics": "half-open [t0-window,t0) vs [t0,t0+window)",
        "window_seconds": args.window_seconds,
    }
    results = {}
    for dataset in ("gaia", "re2ob"):
        manifest_directory = Path(args.manifest_root) / dataset
        split_directory = Path(args.split_root) / dataset
        (
            dataset_manifest,
            split_manifest,
            inputs,
            labels,
            assignments,
        ) = _load_split(manifest_directory, split_directory)
        if dataset == "gaia":
            extraction = _extract_gaia(
                inputs,
                manifest_directory,
                args.window_seconds * 1000,
                args.min_samples_per_side,
                args.score_cap,
                args.top_k_features,
                args.max_gaia_series,
                args.progress_every,
            )
        else:
            extraction = _extract_re2ob(
                inputs,
                manifest_directory,
                args.window_seconds * 1000,
                args.min_samples_per_side,
                args.score_cap,
                args.top_k_features,
                args.max_re2ob_cases,
                args.progress_every,
            )
        rankings, details, predictions, extraction_audit = extraction
        metrics = dict(evaluate_ranking_report(inputs, labels, rankings))
        metrics["coverage_slices"] = _coverage_slice_reports(
            inputs, labels, rankings, predictions
        )
        coverage = _coverage_audit(predictions, inputs[0].services)
        training_audit = {
            "coverage": coverage,
            "extraction": extraction_audit,
            "fit_scope": "none; every score uses only its own case window",
        }
        split_by_case = {row.case_id: row.split for row in assignments}
        result = _write_baseline_result(
            Path(args.output_root) / dataset / "metric_change",
            "metric_change",
            config,
            rankings,
            metrics,
            training_audit,
            split_by_case,
            dataset_manifest,
            manifest_directory,
            split_manifest,
            split_directory,
            prediction_details=details,
        )
        results[dataset] = _compact_result(
            metrics, result["run_manifest_sha256"]
        )

    summary_path = Path(args.summary_output)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
            raise ValueError("existing baseline summary uses another schema")
    else:
        summary = {
            "baseline_schema_version": BASELINE_SCHEMA_VERSION,
            "datasets": {"gaia": {}, "re2ob": {}},
        }
    for dataset, result in results.items():
        summary["datasets"].setdefault(dataset, {})["metric_change"] = result
    summary["metric_change_config"] = config
    _write_json(summary_path, summary)
    return {"config": config, "datasets": results}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--window-seconds", default=300, type=int)
    parser.add_argument("--min-samples-per-side", default=2, type=int)
    parser.add_argument("--top-k-features", default=5, type=int)
    parser.add_argument("--score-cap", default=20.0, type=float)
    parser.add_argument("--max-gaia-series", type=int)
    parser.add_argument("--max-re2ob-cases", type=int)
    parser.add_argument("--progress-every", default=100, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_metric_change(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
