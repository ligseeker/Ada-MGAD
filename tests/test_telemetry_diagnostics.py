from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.schema import RCACaseInput
from src.data.telemetry_diagnostics import (
    ModalityAccumulator,
    TimestampAccumulator,
    _case_window_activity,
    _gaia_log_timestamp_ms,
    _layout_inventory,
    _numeric_timestamp_ms,
    _scan_re2_event_stream,
    _scan_re2_metrics,
)


class TimestampAccumulatorTest(unittest.TestCase):
    def test_sequence_quality_and_file_boundaries(self):
        stats = TimestampAccumulator()
        stats.begin_file()
        stats.update(np.asarray([1000, 4000, 4000], dtype=np.int64), invalid_rows=1)
        stats.update(np.asarray([7000], dtype=np.int64))
        stats.begin_file()
        stats.update(np.asarray([100], dtype=np.int64))

        record = stats.to_record()
        self.assertEqual(record["valid_timestamp_rows"], 5)
        self.assertEqual(record["invalid_timestamp_rows"], 1)
        self.assertEqual(record["positive_delta_rows"], 2)
        self.assertEqual(record["zero_delta_rows"], 1)
        self.assertEqual(record["negative_delta_rows"], 0)
        self.assertEqual(record["timestamp_min_ms"], 100)
        self.assertEqual(record["timestamp_max_ms"], 7000)

    def test_numeric_units_and_invalid_values(self):
        milliseconds, invalid = _numeric_timestamp_ms(
            pd.Series([1, "2", "bad"]), "s"
        )
        self.assertEqual(milliseconds.tolist(), [1000, 2000])
        self.assertEqual(invalid, 1)

    def test_gaia_log_timestamp_comes_from_message_prefix(self):
        messages = pd.Series(
            [
                "2021-07-01 10:54:22,639 | INFO | body",
                "not timestamped",
            ]
        )
        milliseconds, invalid = _gaia_log_timestamp_ms(messages)
        expected = int(
            pd.Timestamp(
                "2021-07-01 10:54:22.639", tz="Asia/Shanghai"
            ).timestamp()
            * 1000
        )
        self.assertEqual(milliseconds.tolist(), [expected])
        self.assertEqual(invalid, 1)


class CoverageSemanticsTest(unittest.TestCase):
    def test_window_activity_separates_bins_and_service_presence(self):
        case = RCACaseInput(
            case_id="case",
            dataset="toy",
            anchor_time=60_000,
            services=("a", "b"),
        )
        modality = ModalityAccumulator("logs")
        modality.timestamps.update(np.asarray([30_000, 60_000, 90_000]))
        modality.service_stats("a").update(np.asarray([60_000]))

        result = _case_window_activity((case,), modality, (30,))
        self.assertEqual(result["30"]["case_any_activity_ratio"], 1.0)
        self.assertEqual(result["30"]["occupied_bin_ratio"]["median"], 1.0)
        self.assertEqual(
            result["30"]["candidate_service_presence_ratio"]["median"], 0.5
        )

    def test_layout_digest_is_order_independent_and_size_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.csv"
            second = root / "b.csv"
            first.write_text("a\n", encoding="utf-8")
            second.write_text("bb\n", encoding="utf-8")
            left = _layout_inventory((first, second), root)
            right = _layout_inventory((second, first), root)
            self.assertEqual(left, right)
            self.assertIn("relative_path", left["digest_scope"])


class RCAEvalStreamTest(unittest.TestCase):
    def test_metric_missingness_and_candidate_service_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "time,a_cpu,b_cpu,host_metric\n"
                "1,1,,3\n"
                "2,2,4,5\n"
                "2,3,5,6\n"
                "4,4,6,7\n",
                encoding="utf-8",
            )
            accumulator, extra = _scan_re2_metrics(
                path, ("a", "b"), chunk_rows=1
            )
            self.assertEqual(accumulator.rows, 4)
            self.assertEqual(accumulator.value_cells, 12)
            self.assertEqual(accumulator.missing_value_cells, 1)
            scheduled = accumulator.to_record()["scheduled_timestamp_quality"]
            self.assertEqual(scheduled["expected_timestamps"], 4)
            self.assertEqual(scheduled["observed_unique_timestamps"], 3)
            self.assertEqual(scheduled["missing_timestamps"], 1)
            self.assertEqual(scheduled["duplicate_timestamp_rows"], 1)
            self.assertEqual(extra["candidate_service_metric_columns"]["a"], 1)
            self.assertEqual(accumulator.service_stats("a").valid_rows, 4)
            self.assertEqual(accumulator.service_stats("b").valid_rows, 3)

    def test_event_required_field_and_timestamp_parse_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.csv"
            path.write_text(
                "timestamp,container_name\n"
                "1000000,a\n"
                "bad,\n",
                encoding="utf-8",
            )
            accumulator, extra = _scan_re2_event_stream(
                path,
                "logs",
                "timestamp",
                "ns",
                "container_name",
                (),
                chunk_rows=1,
            )
            self.assertEqual(accumulator.rows, 2)
            self.assertEqual(accumulator.timestamps.valid_rows, 1)
            self.assertEqual(accumulator.timestamps.invalid_rows, 1)
            self.assertEqual(
                accumulator.required_missing["container_name"], 1
            )
            self.assertEqual(extra["service_entities"], ["a"])


if __name__ == "__main__":
    unittest.main()
