"""The focused C1 Metric audit must replay the adapter's candidate gate."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.p6 import audit_c1_fold1_metric as audit
from scripts.p6.audit_c1_fold1_metric import _check, _run, analyze_metric_candidates
from src.e2e.gaia_preprocessing.raw import fit_metric


START = 1_625_101_200_000
END = START + 4 * 30_000


def _write_metric(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp", "value"))
        writer.writerows((START + step * 30_000, value) for step, value in enumerate(values))


class C1Fold1MetricAuditTest(unittest.TestCase):
    def test_historical_metric_helper_hash_drift_is_rejected(self):
        actual_sha256 = audit._sha256

        def drifted(path):
            if Path(path).name == "metric.py":
                return "0" * 64
            return actual_sha256(path)

        with patch.object(audit, "_sha256", side_effect=drifted):
            with self.assertRaisesRegex(ValueError, "failure source"):
                audit._failure_binding()

    def test_quality_and_redundancy_ledger_matches_actual_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_dir = root / "metric"
            metric_dir.mkdir()
            for service in ("dbservice1", "dbservice2"):
                for feature, values in (
                    ("signal_a", (1, 2, 3, 4)),
                    ("signal_duplicate", (1, 2, 3, 4)),
                    ("signal_flat", (1, 1, 1, 1)),
                ):
                    _write_metric(metric_dir / "{}_0.0.0.1_{}_2021-07-01_2021-07-15.csv".format(
                        service, feature), values)
            _write_metric(metric_dir / "system_0.0.0.1_host_signal_2021-07-01_2021-07-15.csv",
                          (1, 4, 2, 8))
            serial_temp = root / "serial"
            parallel_temp = root / "parallel"
            serial_temp.mkdir()
            parallel_temp.mkdir()
            kwargs = {"grid_ms": 30_000, "pearson_threshold": 0.995,
                      "spearman_threshold": 0.995}
            serial_rows, serial = analyze_metric_candidates(
                metric_dir, (START, END), workers=1, temporary=serial_temp, **kwargs)
            parallel_rows, parallel = analyze_metric_candidates(
                metric_dir, (START, END), workers=2, temporary=parallel_temp, **kwargs)
            self.assertEqual(serial_rows, parallel_rows)
            self.assertEqual(serial["surviving_slot_names"], parallel["surviving_slot_names"])
            self.assertEqual(serial["candidate_tasks"], 4)
            self.assertEqual(serial["quality_pass"], 3)
            self.assertEqual(serial["quality_rejections"], {"unique": 1})
            self.assertEqual(serial["correlation_reject"], 1)
            self.assertEqual(serial["post_correlation_real_slots"], 2)
            adapter = fit_metric(metric_dir, START, END, grid_ms=30_000,
                                 required_slots=2, max_slots=2,
                                 pearson_threshold=0.995, spearman_threshold=0.995,
                                 scope_quotas={"global": 30, "host": 15})
            self.assertEqual(set(serial["surviving_slot_names"]), set(adapter.slot_names))
            self.assertTrue(any(row["redundant_with"] for row in serial_rows))

    def test_isolated_audit_seals_a_27_slot_replay_without_touching_failed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_dir = root / "metric"
            metric_dir.mkdir()
            output = root / "isolated-audit"
            rng = np.random.default_rng(42)
            for index in range(27):
                values = rng.normal(loc=10.0, scale=2.0, size=60)
                _write_metric(metric_dir / "system_0.0.0.1_audit_{:02d}_2021-07-01_2021-07-15.csv".format(index),
                              values)
            protocol = {"resource_budget": {"ad_preprocessing_workers": 24}}
            policy = {"base_slots": 45, "pearson_threshold": 0.995,
                      "spearman_threshold": 0.995}
            bound_raw = {"manifest_sha256": "isolated_fixture", "file_count": 27}
            inputs = (output, protocol, policy, root, root / "unused_manifest.json",
                      metric_dir, (START, START + 60 * 30_000), 30_000)
            with patch("scripts.p6.audit_c1_fold1_metric._inputs", return_value=inputs), \
                 patch("scripts.p6.audit_c1_fold1_metric.verify_raw_content", return_value=bound_raw):
                _run("isolated-audit", 1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "completion_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["reproduces_original_27"])
            self.assertEqual(summary["post_correlation_real_slots"], 27)
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse((output / "INCOMPLETE.json").exists())
            self.assertEqual(set(manifest["files"]), {"candidate_ledger.csv", "summary.json"})
            with patch("scripts.p6.audit_c1_fold1_metric.AUDIT_PARENT", root):
                _check("isolated-audit")
                with (output / "candidate_ledger.csv").open("a", encoding="utf-8") as stream:
                    stream.write("tampered\n")
                with self.assertRaisesRegex(ValueError, "output drift"):
                    _check("isolated-audit")


if __name__ == "__main__":
    unittest.main()
