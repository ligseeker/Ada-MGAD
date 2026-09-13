import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.e2e.ad_data import build_registry_node_labels
from src.e2e.gt import FAULT_TYPES, build_registry, derive_raw_record
from src.e2e.protocol import TemporalBlock, load_config, temporal_blocks


ROOT = Path(__file__).resolve().parents[1]


def raw_row(message, service="dbservice1"):
    return {"datetime": "2021-07-01", "service": service, "message": message}


class V3RawTaxonomyTests(unittest.TestCase):
    def test_six_class_taxonomy_and_float_duration(self):
        records = [
            derive_raw_record(raw_row(
                "2021-07-01 08:00:00,123 | WARNING | n | c | dbservice1 | "
                "[cpu_anomalies] start at 2021-07-01 08:00:01.100 lasts 3.2413992881774902 seconds"
            ), 0),
            derive_raw_record(raw_row(
                "2021-07-01 08:01:00,123 | WARNING | n | c | dbservice1 | "
                "[normal memory freed label] lasts ten minutes"
            ), 1),
            derive_raw_record(raw_row(
                "2021-07-01 08:02:00,123 | ERROR | n | c | dbservice1 | upload failed"
            ), 2),
            derive_raw_record(raw_row(
                "2021-07-01 08:03:00,123 | ERROR | n | c | dbservice1 | "
                "(Background on this error at https://example.invalid)"
            ), 3),
        ]
        self.assertEqual(records[0]["fault_type"], "cpu_anomalies")
        self.assertAlmostEqual(records[0]["duration_seconds"], 3.2413992881774902)
        self.assertEqual(records[1]["fault_type"], "normal_memory_freed")
        self.assertEqual(records[1]["duration_seconds"], 600.0)
        self.assertFalse(records[2]["gt_included"])
        self.assertFalse(records[3]["gt_included"])
        self.assertEqual(records[2]["raw_type"], "error_record")
        self.assertEqual(records[3]["raw_type"], "traceback_continuation")
        self.assertEqual(set(FAULT_TYPES), {
            "login_failure", "memory_anomalies", "file_moving",
            "normal_memory_freed", "access_permission_denied", "cpu_anomalies",
        })

    def test_registry_keeps_boundary_and_domain_purges_without_merging(self):
        records = [
            derive_raw_record(raw_row(
                "2021-07-01 08:00:00,000 | WARNING | n | c | dbservice1 | "
                "[login failure] wait for 11 seconds"
            ), 0),
            derive_raw_record(raw_row(
                "2021-07-01 08:00:00,000 | WARNING | n | c | dbservice1 | "
                "[login failure] wait for 11 seconds"
            ), 1),
        ]
        raw, assigned, purged = build_registry(
            records, detector_start_ms=1_625_090_000_000,
            detector_end_ms=1_625_100_000_000, split_ms=1_625_099_000_000,
        )
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(assigned), 2)
        self.assertEqual(len(purged), 0)
        self.assertEqual(raw["case_id"].nunique(), 2)


class V3SplitAndLabelTests(unittest.TestCase):
    def test_metric_70_30_integer_boundary_has_two_blocks(self):
        config = load_config(ROOT / "configs/e2e/gaia_p5_v3.json")
        blocks = temporal_blocks(config)
        self.assertEqual([block.name for block in blocks], ["train", "test"])
        n = (blocks[-1].end_ms - blocks[0].start_ms) // 30_000
        self.assertEqual(blocks[0].end_ms, blocks[0].start_ms + ((7 * n) // 10) * 30_000)
        self.assertEqual(blocks[0].end_ms, config["split"]["boundary_ms"])

    def test_half_open_overlap_does_not_label_touching_bins(self):
        timestamps = np.asarray([0, 30_000, 60_000, 90_000], dtype=np.int64)
        registry = pd.DataFrame([{
            "case_id": "exact", "service": "dbservice1", "start_ms": 30_000, "end_ms": 60_000,
        }])
        labels = build_registry_node_labels(timestamps, registry)
        self.assertEqual(labels[:, 0].tolist(), [0, 1, 0, 0])

    def test_split_crossing_interval_is_purged_and_context_is_half_open(self):
        blocks = (TemporalBlock("train", 0, 100), TemporalBlock("test", 100, 200, is_final=True))
        frame = pd.DataFrame([{
            "case_id": "cross", "source_index": 0, "service": "dbservice1",
            "fault_type": "login_failure", "start_ms": 99, "end_ms": 101,
        }])
        from src.e2e.protocol import assign_event_blocks
        assigned, purged = assign_event_blocks(frame, blocks)
        self.assertEqual(len(assigned), 0)
        self.assertEqual(purged.iloc[0]["case_id"], "cross")


if __name__ == "__main__":
    unittest.main()
