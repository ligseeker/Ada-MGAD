"""C1 selection labels use only pre-Generation onsets and complete metric GT."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.e2e.c1_fold_supervision import _build_from_registry


class C1FoldSupervisionTest(unittest.TestCase):
    def test_cross_boundary_fault_affects_labels_but_not_selection_metric_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, start in (("fit", 0), ("selection", 600000)):
                directory = root / "ad_data" / name
                directory.mkdir(parents=True)
                np.save(directory / "timestamps.npy", start + np.arange(20, dtype=np.int64) * 30000)
            manifest = {"fold": 1, "segments": {
                "fit": {"interval_ms": [0, 600000]},
                "selection": {"interval_ms": [600000, 1200000]},
                "generation": {"interval_ms": [1200000, 1800000]},
            }}
            registry = pd.DataFrame([
                {"case_id": "fit", "source_index": 1, "service": "dbservice1", "fault_type": "x",
                 "start_ms": 300000, "end_ms": 390000, "detector_domain": True},
                {"case_id": "cross", "source_index": 2, "service": "dbservice1", "fault_type": "x",
                 "start_ms": 570000, "end_ms": 690000, "detector_domain": True},
                {"case_id": "select", "source_index": 3, "service": "mobservice1", "fault_type": "y",
                 "start_ms": 900000, "end_ms": 960000, "detector_domain": True},
                {"case_id": "generation", "source_index": 4, "service": "webservice1", "fault_type": "z",
                 "start_ms": 1250000, "end_ms": 1320000, "detector_domain": True},
            ])
            result = _build_from_registry(root, manifest, registry)
            self.assertEqual(result.selection_gt["case_id"].tolist(), ["select"])
            self.assertEqual(int(result.selection_labels[0]), 1)  # 630s: recent cross-boundary onset
            self.assertEqual(int(result.selection_labels[1]), 2)  # 660s: ongoing fault, ignored
            self.assertEqual(int(result.selection_labels[9]), 1)  # 900s: selection onset
            self.assertEqual(result.audit["selection_cross_boundary_excluded_from_metrics"], 1)
            self.assertFalse(result.audit["generation_labels_built"])
            self.assertFalse(hasattr(result, "generation_labels"))


if __name__ == "__main__":
    unittest.main()
