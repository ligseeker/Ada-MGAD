import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.e2e.gaia_preprocessing.raw import (
    GAIA_SERVICES,
    fit_logs,
    fit_metric,
    fit_trace,
    transform_logs,
    transform_metric,
    transform_trace,
)


START = 1_625_101_200_000  # 2021-07-01 09:00:00 Asia/Shanghai
END = START + 4 * 30_000


def _metric_file(root, name, rows):
    path = root / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        writer.writerows(rows)


def _write_log(path, messages):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "service", "message"])
        for message in messages:
            writer.writerow(["2021-07-01", path.stem, message])


class RawMetricAdapterTests(unittest.TestCase):
    def test_real_slots_are_selected_without_padding_and_counter_is_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [[START + index * 30_000, 10.0 + index * 10.0] for index in range(4)]
            for service in ("dbservice1", "dbservice2"):
                _metric_file(root, "{}_0.0.0.1_docker_cpu_core_0_ticks_2021-07-01_2021-07-15.csv".format(service), rows)
                _metric_file(root, "{}_0.0.0.1_docker_cpu_core_1_ticks_2021-07-01_2021-07-15.csv".format(service), rows)
                _metric_file(root, "{}_0.0.0.1_docker_cpu_total_pct_2021-07-01_2021-07-15.csv".format(service), rows)
                _metric_file(root, "{}_0.0.0.1_docker_memory_usage_pct_2021-07-01_2021-07-15.csv".format(service), rows)

            fitted = fit_metric(root, START, END, min_coverage=0.5, min_unique=2,
                                min_dynamic_ratio=0.0, required_slots=1, max_slots=8)
            self.assertTrue(fitted.slots)
            self.assertTrue(all("padding" not in name for name in fitted.slot_names))
            self.assertTrue(any("service_type_db" in name for name in fitted.slot_names))
            transformed = transform_metric(fitted, root, START, END)
            self.assertEqual(transformed.shape[:2], (4, 10))
            self.assertEqual(transformed.shape[-1], len(fitted.slots))
            self.assertTrue(np.isfinite(transformed).all())

            reduced = fit_metric(root, START, END, min_coverage=0.5, min_unique=2,
                                 min_dynamic_ratio=0.0, required_slots=1,
                                 pearson_threshold=0.99, spearman_threshold=0.99)
            gauge_names = [name for name in reduced.slot_names if "docker_" in name]
            self.assertEqual(len(gauge_names), 1)

    def test_scope_quota_overflow_uses_a_real_slot_to_meet_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [[START + index * 30_000, float(index + 1)] for index in range(4)]
            for service in ("dbservice1", "dbservice2"):
                _metric_file(root, "{}_0.0.0.1_db_signal_2021-07-01_2021-07-15.csv".format(service), rows)
            _metric_file(root, "system_0.0.0.4_host_signal_2021-07-01_2021-07-15.csv", rows)
            fitted = fit_metric(root, START, END, min_coverage=0.5, min_unique=2,
                                min_dynamic_ratio=0.0, required_slots=2, max_slots=2,
                                scope_quotas={"service_type": 1, "host": 0})
            self.assertEqual(len(fitted.slots), 2)
            self.assertEqual({slot.scope for slot in fitted.slots}, {"service_type_db", "host"})

    def test_metric_budget_fails_closed_when_real_slots_are_insufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [[START + index * 30_000, float(index)] for index in range(4)]
            for service in ("dbservice1", "dbservice2"):
                _metric_file(root, "{}_0.0.0.1_only_2021-07-01_2021-07-15.csv".format(service), rows)
            with self.assertRaisesRegex(ValueError, "below required budget"):
                fit_metric(root, START, END, min_coverage=0.5, min_unique=2,
                           min_dynamic_ratio=0.0, required_slots=2)


class RawLogAdapterTests(unittest.TestCase):
    def test_level_aware_routes_and_frozen_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stable = "2021-07-01 09:00:00,000 | INFO | x | stable message"
            error = "2021-07-01 09:00:30,000 | ERROR | x | unique error {}"
            for service in GAIA_SERVICES:
                messages = [stable, stable]
                if service == "dbservice1":
                    messages.append(error.format(service))
                _write_log(root / "business_table_{}_2021-07.csv".format(service), messages)
            fitted = fit_logs(root, START, END, min_template_count=2, min_template_bins=1)
            self.assertTrue(fitted.slot_names)
            self.assertIn("RARE_ERROR", fitted.slot_names)
            self.assertIn("UNK_ERROR", fitted.slot_names)
            transformed = transform_logs(fitted, root, START, END)
            self.assertEqual(transformed.shape[:2], (4, 10))
            self.assertEqual(transformed.shape[-1], len(fitted.slot_names))
            self.assertTrue(np.isfinite(transformed).all())


class RawTraceAdapterTests(unittest.TestCase):
    def test_split_local_parent_and_count_mean_latency_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = ["timestamp", "host_ip", "service_name", "trace_id", "span_id", "parent_id", "start_time", "end_time", "url", "status_code", "message"]
            for service in GAIA_SERVICES:
                path = root / "trace_table_{}_2021-07.csv".format(service)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    if service == "dbservice1":
                        writer.writerow(["2021-07-01 09:00:00", "0.0.0.4", service, "t", "p", "0", "2021-07-01 09:00:00.000", "2021-07-01 09:00:00.010", "", "200", "root"])
                    if service == "webservice1":
                        writer.writerow(["2021-07-01 09:00:00", "0.0.0.1", service, "t", "c", "p", "2021-07-01 09:00:00.020", "2021-07-01 09:00:00.120", "", "500", "child"])
            fitted = fit_trace(root, START, END, min_edge_rows=1, min_positive_bins=1)
            self.assertEqual(fitted.directed_edges, (("dbservice1", "webservice1"),))
            transformed = transform_trace(fitted, root, START, END)
            source = GAIA_SERVICES.index("dbservice1")
            destination = GAIA_SERVICES.index("webservice1")
            self.assertEqual(transformed.shape, (4, 10, 10, 8))
            self.assertGreater(float(transformed[0, source, destination, 6]), 0.0)
            self.assertGreater(float(transformed[0, source, destination, 7]), 0.0)


if __name__ == "__main__":
    unittest.main()
