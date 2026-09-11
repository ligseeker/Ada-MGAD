import unittest
from pathlib import Path

import pandas as pd

from src.e2e.protocol import (
    TemporalBlock,
    assign_event_blocks,
    layout_digest,
    load_config,
    load_registry,
    purge_rca_cases,
    temporal_blocks,
)
from util.GAIA.pre_GAIA import parse_anomaly_event


ROOT = Path(__file__).resolve().parents[1]


def event(message, service="dbservice1", declared="2021-07-01"):
    return parse_anomaly_event(pd.Series({
        "message": message,
        "service": service,
        "datetime": declared,
    }))


class EventParserTests(unittest.TestCase):
    def test_cpu_float_duration(self):
        parsed = event(
            "2021-07-01 10:00:00,000 | WARNING | [cpu_anomalies] "
            "start at 2021-07-01 10:00:01.100 lasts 3.2413992881774902 seconds"
        )
        self.assertEqual(parsed["anomaly_type"], "[cpu_anomalies]")
        self.assertAlmostEqual(parsed["duration"], 3.2413992881774902)
        self.assertTrue(parsed["st_time"])
        self.assertTrue(parsed["ed_time"])

    def test_supported_event_semantics_unchanged(self):
        messages = {
            "[login failure]": "2021-07-01 10:00:00,123 | WARNING | simulate the login failure wait for 11 seconds",
            "[memory_anomalies]": "2021-07-01 10:00:00,000 | WARNING | [memory_anomalies] start at 2021-07-01 10:00:01.100 lasts 600 seconds",
            "[file moving program]": "2021-07-01 10:00:00,000 | WARNING | trigger the file moving program start with 2021-07-01 10:00:01.100 last for 60 seconds",
            "[access permission denied exception]": "2021-07-01 10:00:00,123 | WARNING | access permission denied exception lasts 30 seconds",
        }
        for expected, message in messages.items():
            with self.subTest(expected=expected):
                parsed = event(message)
                self.assertEqual(parsed["anomaly_type"], expected)
                self.assertTrue(parsed["st_time"])
                self.assertTrue(parsed["ed_time"])


class TemporalProtocolTests(unittest.TestCase):
    def test_layout_digest_binds_relative_paths_and_sizes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.csv").write_bytes(b"12")
            (root / "a.csv").write_bytes(b"345")
            result = layout_digest(root, root.glob("*.csv"))
            self.assertEqual(result["files"], 2)
            self.assertEqual(result["bytes"], 5)
            self.assertEqual(len(result["layout_sha256"]), 64)

    def test_frozen_registry_and_ordered_blocks(self):
        config = load_config(ROOT / "configs/e2e/gaia_p5_v1.yaml")
        registry = load_registry(config, ROOT)
        blocks = temporal_blocks(config)
        self.assertEqual(len(registry), 16200)
        self.assertLess(blocks[0].start_ms, blocks[0].end_ms)
        self.assertEqual(blocks[0].end_ms, blocks[1].start_ms)
        self.assertEqual(blocks[1].end_ms, blocks[2].start_ms)

    def test_crossing_injections_and_contexts_are_purged(self):
        blocks = (
            TemporalBlock("train", 0, 10000),
            TemporalBlock("validation", 10000, 20000),
            TemporalBlock("test", 20000, 30000, is_final=True),
        )
        frame = pd.DataFrame([
            {"case_id": "a", "source_index": 0, "service": "dbservice1", "fault_type": "memory_anomalies", "start_ms": 2000, "end_ms": 2100},
            {"case_id": "cross", "source_index": 1, "service": "dbservice1", "fault_type": "memory_anomalies", "start_ms": 9900, "end_ms": 10100},
            {"case_id": "edge", "source_index": 2, "service": "dbservice1", "fault_type": "memory_anomalies", "start_ms": 9500, "end_ms": 9600},
        ])
        assigned, raw_purged = assign_event_blocks(frame, blocks)
        self.assertEqual(set(raw_purged["case_id"]), {"cross"})
        retained, context_purged = purge_rca_cases(assigned, blocks, 1)
        self.assertEqual(set(retained["case_id"]), {"a"})
        self.assertEqual(set(context_purged["case_id"]), {"edge"})


if __name__ == "__main__":
    unittest.main()
