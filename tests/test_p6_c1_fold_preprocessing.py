"""G3 fold input contracts on isolated synthetic timelines and raw files."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.e2e import c1_fold_preprocessing as c1
from src.e2e.protocol import GAIA_SERVICES


PROTOCOL = c1.ROOT / "configs/e2e/gaia_p6_c1_g2_v1_1.json"
SYNTHETIC_INTERVALS = {"fit": (0, 600000), "selection": (600000, 1200000),
                       "generation": (1200000, 1800000)}


def _fake_fits(stable_count=17):
    scaler = SimpleNamespace(as_dict=lambda: {"lower": 0.0, "upper": 1.0})
    slots = tuple(SimpleNamespace(
        name="metric_{:02d}".format(index), scope="global",
        logical_feature="metric_{:02d}".format(index), statistic="value",
        semantic_kind="gauge", targets=GAIA_SERVICES,
        source_keys=("metric_{:02d}".format(index),)) for index in range(45))
    metric = SimpleNamespace(slots=slots, slot_names=tuple(slot.name for slot in slots),
                             scalers={slot.name: scaler for slot in slots}, fill_max_age_ms=60000)
    names = tuple("stable_template_{}".format(index) for index in range(stable_count))
    for kind in ("RARE", "UNK", "level"):
        names += tuple("{}_{}".format(kind, level) for level in
                       ("INFO", "WARNING", "ERROR", "DEBUG", "UNKNOWN"))
    logs = SimpleNamespace(stable_cluster_ids=tuple(range(stable_count)), slot_names=names,
                           drain_state="{}", scalers={}, cluster_stats={})
    trace = SimpleNamespace(directed_edges=((GAIA_SERVICES[0], GAIA_SERVICES[1]),),
                            scalers={}, diagnostics={})
    return metric, logs, trace


class C1FoldPreprocessingTest(unittest.TestCase):
    def test_locked_fold_geometry_and_history_are_segment_local(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        intervals = c1._fold_intervals(protocol, 1, 30000, 10)
        self.assertEqual(intervals["fit"], (1625133600000, 1625394960000))
        self.assertEqual(intervals["selection"][0], intervals["fit"][1])
        self.assertEqual(intervals["generation"][0], intervals["selection"][1])
        with self.assertRaisesRegex(ValueError, "three locked folds"):
            c1._fold_intervals(protocol, 4, 30000, 10)

    def test_raw_content_check_rejects_changed_bytes_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            root.mkdir()
            path = root / "metric.csv"
            path.write_bytes(b"time,value\n0,1\n")
            record = {"path": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            rows = [record]
            manifest = {"schema_version": "p6_c1_raw_content_inventory_v1",
                        "raw_root": str(root), "file_count": 1,
                        "total_bytes": record["bytes"], "files": rows,
                        "records_sha256": hashlib.sha256(json.dumps(
                            rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(c1.verify_raw_content(manifest_path, root)["file_count"], 1)
            path.write_bytes(b"time,value\n0,2\n")
            with self.assertRaisesRegex(ValueError, "content drift"):
                c1.verify_raw_content(manifest_path, root)
            record["path"] = "../outside.csv"
            manifest["records_sha256"] = hashlib.sha256(json.dumps(
                rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside raw root"):
                c1.verify_raw_content(manifest_path, root)

    def test_synthetic_fold_uses_fit_only_and_seals_label_free_segments(self):
        metric_fit, log_fit, trace_fit = _fake_fits()
        expected_raw = {"file_count": 6661, "total_bytes": 31583130281,
                        "records_sha256": "ee1ce9d50bf4002636223b31045e86c3d7000736700201ff655bb05a74776052",
                        "manifest_sha256": "03bdefc1ad817802a9f015a5cc07c71f5e8d625c66c5d5e9fbff31d0d5671718"}

        def tensor(start, end, shape):
            return np.zeros(((end - start) // 30000,) + shape, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            fold_root = Path(temporary) / "folds/fold_01"
            with patch.object(c1, "verify_raw_content", return_value=expected_raw), \
                 patch.object(c1, "_raw_train_binding", return_value="a" * 64), \
                 patch.object(c1, "_fold_intervals", return_value=SYNTHETIC_INTERVALS), \
                 patch.object(c1, "fit_metric", return_value=metric_fit) as fit_metric, \
                 patch.object(c1, "fit_logs", return_value=log_fit) as fit_logs, \
                 patch.object(c1, "fit_trace", return_value=trace_fit) as fit_trace, \
                 patch.object(c1, "transform_metric_with_observability",
                              side_effect=lambda _, __, start, end, **kw:
                              (tensor(start, end, (10, 48)), None)), \
                 patch.object(c1, "transform_logs",
                              side_effect=lambda _, __, start, end, **kw:
                              tensor(start, end, (10, 32))), \
                 patch.object(c1, "transform_trace_with_diagnostics",
                              side_effect=lambda _, __, start, end, **kw:
                              (tensor(start, end, (10, 10, 8)),
                               {"cross_segment_parent_unmatched": 2})):
                manifest = c1.materialize_c1_fold(
                    protocol_path=PROTOCOL, fold_number=1, fold_root=fold_root,
                    runtime={"workers": 1, "start_method": "spawn"})
            for fit in (fit_metric, fit_logs, fit_trace):
                self.assertEqual(fit.call_args.args[1:3], (0, 600000))
            self.assertEqual(manifest["decision_inputs"], ["fit"])
            self.assertEqual(manifest["dimensions"], c1.EXPECTED_DIMENSIONS)
            self.assertEqual(manifest["segments"]["selection"]["first_prediction_available_ms"], 900000)
            self.assertEqual(manifest["segments"]["selection"]["last_prediction_available_ms"], 1170000)
            self.assertEqual(manifest["segments"]["selection"]["legal_windows"], 10)
            np.testing.assert_array_equal(np.load(fold_root / "ad_data/selection/legal_window_indices.npy"),
                                          np.arange(10))
            self.assertEqual(manifest["segments"]["generation"]["trace_diagnostics"]["cross_segment_parent_unmatched"], 2)
            self.assertFalse(list(fold_root.rglob("labels.npy")))
            with patch.object(c1, "_fold_intervals", return_value=SYNTHETIC_INTERVALS):
                c1.validate_c1_fold(fold_root, protocol_path=PROTOCOL)
            with self.assertRaisesRegex(ValueError, "new folds/fold_XX"):
                c1.materialize_c1_fold(protocol_path=PROTOCOL, fold_number=1,
                                        fold_root=fold_root, runtime={"workers": 1})
            (fold_root / "ad_data/selection/metric.npy").write_bytes(b"tampered")
            with patch.object(c1, "_fold_intervals", return_value=SYNTHETIC_INTERVALS):
                with self.assertRaisesRegex(ValueError, "output drift"):
                    c1.validate_c1_fold(fold_root, protocol_path=PROTOCOL)

    def test_prefix_fit_failure_retains_incomplete_directory(self):
        expected_raw = {"manifest_sha256":
                        "03bdefc1ad817802a9f015a5cc07c71f5e8d625c66c5d5e9fbff31d0d5671718"}
        with tempfile.TemporaryDirectory() as temporary:
            fold_root = Path(temporary) / "folds/fold_01"
            with patch.object(c1, "verify_raw_content", return_value=expected_raw), \
                 patch.object(c1, "_fold_intervals", return_value=SYNTHETIC_INTERVALS), \
                 patch.object(c1, "fit_metric", side_effect=ValueError("qualified real Metric slots below required budget")):
                with self.assertRaisesRegex(ValueError, "qualified real Metric slots"):
                    c1.materialize_c1_fold(protocol_path=PROTOCOL, fold_number=1,
                                            fold_root=fold_root, runtime={"workers": 1})
            self.assertTrue((fold_root / "INCOMPLETE.json").is_file())
            self.assertFalse((fold_root / "completion_manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "new folds/fold_XX"):
                c1.materialize_c1_fold(protocol_path=PROTOCOL, fold_number=1,
                                        fold_root=fold_root, runtime={"workers": 1})


if __name__ == "__main__":
    unittest.main()
