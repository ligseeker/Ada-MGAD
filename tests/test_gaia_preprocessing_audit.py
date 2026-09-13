import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.p5.audit_gaia_multimodal_preprocessing import (
    _correlation_matrices,
    audit_metrics,
    _insert_parent_rows,
    _open_parent_index,
    _parent_lookup,
)


class TraceAuditParentIndexTests(unittest.TestCase):
    def test_cross_service_key_is_ambiguous_but_same_service_duplicate_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = _open_parent_index(Path(temporary) / "parents.sqlite3")
            key_ambiguous = b"a" * 16
            key_unique = b"b" * 16

            stats = _insert_parent_rows(
                connection,
                [
                    (key_ambiguous, "dbservice1"),
                    (key_ambiguous, "redisservice1"),
                    (key_unique, "webservice1"),
                    (key_unique, "webservice1"),
                ],
            )
            resolved, ambiguous = _parent_lookup(
                connection, [key_ambiguous, key_unique]
            )
            connection.close()

            self.assertEqual(resolved, {key_unique: "webservice1"})
            self.assertEqual(ambiguous, {key_ambiguous})
            self.assertEqual(stats["ambiguous_keys_created"], 1)
            self.assertEqual(stats["same_service_duplicates"], 1)


class MetricAuditSemanticTests(unittest.TestCase):
    def test_correlation_uses_only_pairwise_observed_bins(self):
        names, pearson, spearman = _correlation_matrices({
            "left": np.asarray([1.0, 2.0, 3.0, 100.0, np.nan]),
            "right": np.asarray([1.0, 2.0, 3.0, np.nan, -100.0]),
        })

        self.assertEqual(names, ["left", "right"])
        self.assertAlmostEqual(pearson[0, 1], 1.0)
        self.assertAlmostEqual(spearman[0, 1], 1.0)

    def test_counter_quality_is_computed_after_reset_aware_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_root = root / "metric"
            output = root / "output"
            metric_root.mkdir()
            output.mkdir()
            start = 1_625_133_600_000
            pd.DataFrame({
                "timestamp": [start, start + 30_000, start + 60_000, start + 90_000],
                "value": [10.0, 40.0, 5.0, 35.0],
            }).to_csv(
                metric_root
                / "dbservice1_0.0.0.4_docker_network_in_bytes_2021-07-01_2021-07-15.csv",
                index=False,
            )

            result = audit_metrics(
                metric_root, output, start, start + 120_000, chunk_rows=2
            )
            quality = pd.read_csv(output / "metric_quality_train.csv").iloc[0]

            self.assertEqual(quality["semantic_kind"], "counter")
            self.assertEqual(
                quality["transformed_feature"], "docker_network_in_bytes__rate"
            )
            self.assertGreaterEqual(float(quality["q01"]), 0.0)
            self.assertEqual(result["semantic_transform_counts"], {"counter": 1})


if __name__ == "__main__":
    unittest.main()
