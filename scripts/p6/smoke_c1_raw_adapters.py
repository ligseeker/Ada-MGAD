#!/usr/bin/env python3
"""Bounded real-adapter C1 fixture: workers=1 versus frozen workers=24.

This script generates a tiny synthetic raw catalog in a temporary directory.
It never reads the full GAIA corpus or creates a formal C1 fold.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e2e.gaia_preprocessing.raw import (
    fit_logs, fit_metric, fit_trace, transform_logs,
    transform_metric_with_observability, transform_trace_with_diagnostics,
)
from src.e2e.protocol import GAIA_SERVICES


START = 1_625_101_200_000
GRID = 30_000
SEGMENTS = {"fit": (START, START + 64 * GRID),
            "selection": (START + 64 * GRID, START + 84 * GRID),
            "generation": (START + 84 * GRID, START + 104 * GRID)}
WORDS = (
    "amber", "beacon", "cobalt", "delta", "ember", "forest", "galaxy",
    "harbor", "island", "jungle", "kernel", "lantern", "meadow", "nebula",
    "orchid", "prairie", "quartz",
)


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _local_time(ms: int) -> str:
    from datetime import timedelta
    value = datetime.fromtimestamp(ms / 1000.0, tz=timezone(timedelta(hours=8)))
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fixture(root: Path) -> Mapping[str, Path]:
    metric = root / "metric"
    logs = root / "business"
    traces = root / "trace"
    timestamps = START + np.arange(104, dtype=np.int64) * GRID
    for logical in range(30):
        suffix = "{}{}".format(chr(97 + logical // 26), chr(97 + logical % 26))
        for service_index, service in enumerate(GAIA_SERVICES):
            rng = np.random.default_rng(1000 + logical * 10 + service_index)
            values = rng.uniform(5.0, 15.0, len(timestamps))
            name = "{}_0.0.0.1_smokeglobal{}_2021-07-01_2021-07-15.csv".format(service, suffix)
            _write_csv(metric / name, ("timestamp", "value"), zip(timestamps, values))
    for logical in range(15):
        suffix = "{}{}".format(chr(97 + logical // 26), chr(97 + logical % 26))
        rng = np.random.default_rng(5000 + logical)
        values = rng.uniform(5.0, 15.0, len(timestamps))
        name = "system_0.0.0.1_smokehost{}_2021-07-01_2021-07-15.csv".format(suffix)
        _write_csv(metric / name, ("timestamp", "value"), zip(timestamps, values))
    for service in GAIA_SERVICES:
        messages = []
        for index, word in enumerate(WORDS):
            payload = "{} {} {} {}".format(word, word, word, word)
            for bin_index in (index + 1, index + 24, 70, 90):
                message = _local_time(START + bin_index * GRID)[:19].replace(".", ",")
                message += ",000 | INFO | smoke | " + payload
                messages.append((message,))
        _write_csv(logs / "business_table_{}_2021-07.csv".format(service), ("message",), messages)
    trace_header = ("trace_id", "span_id", "parent_id", "service_name",
                    "start_time", "end_time", "status_code")
    for service in GAIA_SERVICES:
        rows = []
        for bin_index in (5, 15, 70, 90):
            trace_id = "smoke-{}".format(bin_index)
            if service == "dbservice1":
                rows.append((trace_id, "parent", "0", service,
                             _local_time(START + bin_index * GRID + 1000),
                             _local_time(START + bin_index * GRID + 2000), "200"))
            if service == "webservice1":
                rows.append((trace_id, "child", "parent", service,
                             _local_time(START + bin_index * GRID + 3000),
                             _local_time(START + bin_index * GRID + 4000), "500"))
        _write_csv(traces / "trace_table_{}_2021-07.csv".format(service), trace_header, rows)
    return {"metric": metric, "logs": logs, "traces": traces}


def _run(paths, workers: int):
    fit_start, fit_end = SEGMENTS["fit"]
    metric = fit_metric(paths["metric"], fit_start, fit_end,
                        grid_ms=GRID, required_slots=45, max_slots=45,
                        fill_max_intervals=2, pearson_threshold=0.995,
                        spearman_threshold=0.995,
                        scope_quotas={"global": 30, "host": 15},
                        workers=workers, start_method="spawn")
    logs = fit_logs(paths["logs"], fit_start, fit_end,
                    grid_ms=GRID, min_template_count=2, min_template_bins=2,
                    max_stable_templates=17, config_path=ROOT / "util/GAIA/gaia.ini",
                    workers=workers, start_method="spawn")
    trace = fit_trace(paths["traces"], fit_start, fit_end,
                      grid_ms=GRID, min_edge_rows=1, min_positive_bins=2,
                      workers=workers, start_method="spawn")
    if (len(metric.slot_names) != 45 or len(logs.slot_names) != 32
            or len(trace.directed_edges) < 1):
        raise ValueError("synthetic prefix failed frozen C1 dimensions/graph")
    arrays = {}
    diagnostics = {}
    for segment, (start, end) in SEGMENTS.items():
        metric_array, _ = transform_metric_with_observability(
            metric, paths["metric"], start, end, workers=workers, start_method="spawn")
        log_array = transform_logs(logs, paths["logs"], start, end,
                                   workers=workers, start_method="spawn")
        trace_array, trace_diagnostics = transform_trace_with_diagnostics(
            trace, paths["traces"], start, end,
            workers=workers, start_method="spawn")
        if (metric_array.shape != ((end - start) // GRID, 10, 48)
                or log_array.shape != ((end - start) // GRID, 10, 32)
                or trace_array.shape != ((end - start) // GRID, 10, 10, 8)):
            raise ValueError("C1 smoke arrays differ from locked dimensions")
        if any(not np.isfinite(array).all() for array in (metric_array, log_array, trace_array)):
            raise ValueError("C1 smoke array is non-finite")
        arrays[segment] = {"metric": metric_array, "logs": log_array, "trace": trace_array}
        diagnostics[segment] = trace_diagnostics
    return metric, logs, trace, arrays, diagnostics


def _compare(left, right):
    lm, ll, lt, la, ld = left
    rm, rl, rt, ra, rd = right
    if (lm.slot_names != rm.slot_names or lm.scalers != rm.scalers
            or ll.slot_names != rl.slot_names or ll.stable_cluster_ids != rl.stable_cluster_ids
            or ll.scalers != rl.scalers or ll.drain_state != rl.drain_state
            or lt.directed_edges != rt.directed_edges or lt.scalers != rt.scalers
            or ld != rd):
        raise ValueError("workers=1/24 fitted state or trace diagnostics differ")
    hashes = {}
    for segment in SEGMENTS:
        hashes[segment] = {}
        for modality in ("metric", "logs", "trace"):
            a, b = la[segment][modality], ra[segment][modality]
            if not np.array_equal(a, b):
                raise ValueError("workers=1/24 {} {} arrays differ".format(segment, modality))
            hashes[segment][modality] = hashlib.sha256(a.tobytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE.json").write_text('{"status":"INCOMPLETE"}\n', encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="p6_c1_g3_raw_smoke_") as temporary:
            paths = _fixture(Path(temporary))
            serial = _run(paths, 1)
            parallel = _run(paths, 24)
            hashes = _compare(serial, parallel)
        record = {"schema_version": "p6_c1_g3_raw_adapter_smoke_v1", "status": "PASS",
                  "fixture": "synthetic_104_bins_45_metric_slots_17_log_templates_one_trace_edge",
                  "workers_compared": [1, 24], "start_method": "spawn",
                  "fit_only_schema": True, "formal_gaia_prefix_tested": False,
                  "array_sha256": hashes,
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        (output / "INCOMPLETE.json").unlink()
        print("PASS C1 bounded raw adapter smoke; workers 1/24 arrays identical")
    except Exception:
        raise


if __name__ == "__main__":
    main()
