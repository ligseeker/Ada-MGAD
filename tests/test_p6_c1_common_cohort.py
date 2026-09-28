"""C1 cohort extracts paired anchors without passing roots into raw features."""

import unittest

import pandas as pd

from src.e2e.c1_common_cohort import build_c1_common_cohort, require_c1_common_cohort_floor
from src.e2e.c1_oos_matching import C1OOSMatching


class _RecordingIndex:
    def __init__(self):
        self.anchors = []

    def case_indicators(self, anchor_ms, spec):
        self.anchors.append((anchor_ms, spec.window_seconds, spec.bin_seconds))
        return {}


class C1CommonCohortTest(unittest.TestCase):
    def test_paired_features_use_only_anchor_times_and_keep_floor_no_go(self):
        matches = []
        for fold in (1, 2, 3):
            candidates = pd.DataFrame([{
                "fold": fold, "case_id": "case-{}".format(fold),
                "prediction_id": "fold_{:02d}-generation-pred-000000".format(fold),
                "gt_start_ms": 600000 + fold * 1000000,
                "t_hat": 630000 + fold * 1000000,
                "gt_service": "dbservice1", "fault_type": "fault",
            }])
            matches.append(C1OOSMatching(
                pd.DataFrame(), candidates, pd.DataFrame(), {"fold": fold}))
        index = _RecordingIndex()
        cohort = build_c1_common_cohort(fold_matchings=matches, index=index)
        self.assertEqual(cohort.gt_features.shape, (3, 10, 68))
        self.assertEqual(cohort.detected_features.shape, (3, 10, 68))
        self.assertEqual(cohort.root_indices.tolist(), [0, 0, 0])
        self.assertEqual(len(index.anchors), 6)
        self.assertEqual(cohort.coverage_by_fold, {1: 1, 2: 1, 3: 1})
        self.assertFalse(cohort.floors_pass)
        with self.assertRaisesRegex(ValueError, "NO_GO"):
            require_c1_common_cohort_floor(cohort)


if __name__ == "__main__":
    unittest.main()
