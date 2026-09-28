"""One GT-fitted scaler is reused by both C1 Conditional Logit arms."""

import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.e2e.c1_shared_rca import fit_c1_shared_rca_arms
from src.e2e.rca_model import fit_conditional_logit


class C1SharedRcaTest(unittest.TestCase):
    def test_supplied_scaler_matches_legacy_gt_fit_and_is_shared_with_detected_arm(self):
        gt = np.zeros((4, 10, 68), dtype=np.float64)
        for case in range(4):
            gt[case, :, 0] = np.arange(10) + case
            gt[case, case, 1] = 3.0
        detected = gt.copy()
        detected[:, :, 0] = detected[:, ::-1, 0]
        roots = np.arange(4, dtype=np.int64)
        scaler = StandardScaler().fit(gt.reshape(-1, 68))
        legacy = fit_conditional_logit(gt, roots, max_iter=20)
        supplied = fit_conditional_logit(
            gt, roots, max_iter=20,
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        np.testing.assert_array_equal(legacy.scaler_mean, supplied.scaler_mean)
        np.testing.assert_array_equal(legacy.scaler_scale, supplied.scaler_scale)
        np.testing.assert_allclose(legacy.weights, supplied.weights, rtol=0, atol=1e-12)
        arms = fit_c1_shared_rca_arms(
            case_ids=("c0", "c1", "c2", "c3"), gt_features=gt,
            detected_features=detected, root_indices=roots)
        np.testing.assert_array_equal(arms.scaler_mean, arms.gt_arm_b.scaler_mean)
        np.testing.assert_array_equal(arms.scaler_mean, arms.detected_arm_c.scaler_mean)
        np.testing.assert_array_equal(arms.scaler_scale, arms.detected_arm_c.scaler_scale)
        self.assertEqual(arms.gt_arm_b.train_case_indices, arms.detected_arm_c.train_case_indices)
        with self.assertRaisesRegex(ValueError, "together"):
            fit_conditional_logit(gt, roots, scaler_mean=scaler.mean_)


if __name__ == "__main__":
    unittest.main()
