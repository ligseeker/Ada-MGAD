import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.e2e.gaia_rca_adapter import (
    GaiaRcaRawIndex,
    IndexedSeries,
    binned_mean,
    build_raw_index,
    parse_trace_chunk,
)


SPEC = SimpleNamespace(window_seconds=300, bin_seconds=15, n_bins=40)


class TraceAdapterTests(unittest.TestCase):
    def test_status_not_200_latency_and_timestamp_units(self):
        frame = pd.DataFrame({
            "start_time": [
                "2021-07-01 10:00:00.000000",
                "2021-07-01 10:00:01.000000",
                "2021-07-01 10:00:02.000000",
                "2021-07-01 10:00:03.000000",
            ],
            "end_time": [
                "2021-07-01 10:00:00.250000",
                "2021-07-01 10:00:01.500000",
                "2021-07-01 10:00:02.750000",
                "2021-07-01 10:00:04.000000",
            ],
            "status_code": [200, 300, 400, 500],
            "service_name": ["dbservice1"] * 4,
        })
        arrays, stats = parse_trace_chunk(frame)
        self.assertEqual(arrays["trace_error"].tolist(), [False, True, True, True])
        np.testing.assert_allclose(arrays["latency_seconds"], [0.25, 0.5, 0.75, 1.0])
        expected = int(pd.Timestamp(
            "2021-07-01 10:00:00.250", tz="Asia/Shanghai"
        ).timestamp() * 1000)
        self.assertEqual(int(arrays["timestamp_ms"][0]), expected)
        self.assertEqual(stats["trace_error_rows"], 3)
        self.assertEqual(stats["negative_latency_rows"], 0)

    def test_negative_and_invalid_latency_are_filtered(self):
        frame = pd.DataFrame({
            "start_time": ["2021-07-01 10:00:02.0", "bad"],
            "end_time": ["2021-07-01 10:00:01.0", "bad"],
            "status_code": [500, 500],
            "service_name": ["dbservice1", "dbservice1"],
        })
        arrays, stats = parse_trace_chunk(frame)
        self.assertEqual(len(arrays["timestamp_ms"]), 0)
        self.assertEqual(stats["negative_latency_rows"], 1)
        self.assertEqual(stats["invalid_latency_rows"], 1)


class EventRelativeIndexTests(unittest.TestCase):
    def test_exact_anchor_half_open_binning(self):
        anchor = 1_000_000
        start = anchor - 300_000
        timestamps = np.asarray([start - 1, start, start + 14_999, start + 15_000, start + 600_000])
        values = np.asarray([99.0, 1.0, 3.0, 5.0, 99.0])
        result = binned_mean(timestamps, values, anchor, SPEC)
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[1], 5.0)
        self.assertTrue(np.isnan(result[-1]))

    def test_four_channels_and_no_detector_feature(self):
        anchor = 1_000_000
        timestamps = np.asarray([anchor - 1000, anchor + 1000], dtype=np.int64)
        index = GaiaRcaRawIndex(
            [IndexedSeries("dbservice1", "cpu", timestamps, np.asarray([1.0, 3.0]))],
            {"dbservice1": (timestamps, np.asarray([0, 2], dtype=np.uint8))},
            {"dbservice1": [(timestamps, np.asarray([0, 1]), np.asarray([0.1, 0.2]))]},
        )
        result = index.case_indicators(anchor, SPEC)
        self.assertEqual(set(result), {"metric", "log", "trace-error", "trace-latency"})
        self.assertNotIn("detector", " ".join(
            key for channel in result.values() for key in channel
        ))


class LabelFirewallTests(unittest.TestCase):
    def test_raw_index_builder_accepts_no_label_inputs(self):
        parameters = set(inspect.signature(build_raw_index).parameters)
        forbidden = {
            name for name in parameters
            if name in {"root", "root_service", "fault", "fault_type", "label", "labels"}
            or "fault" in name or "label" in name
        }
        self.assertEqual(forbidden, set())


if __name__ == "__main__":
    unittest.main()
