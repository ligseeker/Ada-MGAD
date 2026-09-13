import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.p5.run_i1_rca import train_v3
from scripts.p5.run_i1_rca_features import build_case_registry, materialize
from src.e2e.protocol import GAIA_SERVICES, load_config, temporal_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/e2e/gaia_p5_v3.json"


def _v3_config():
    return load_config(CONFIG_PATH)


def _registry(config):
    blocks = temporal_blocks(config)
    train_anchor = blocks[0].start_ms + 600_000
    test_anchor = blocks[1].start_ms + 600_000
    return pd.DataFrame([
        {
            "case_id": "train-0", "source_index": 0, "service": "dbservice1",
            "labelled_service": "dbservice1", "fault_type": "login_failure",
            "start_ms": train_anchor, "end_ms": train_anchor + 30_000,
        },
        {
            "case_id": "test-keep", "source_index": 1, "service": "dbservice2",
            "labelled_service": "dbservice2", "fault_type": "memory_anomalies",
            "start_ms": test_anchor, "end_ms": test_anchor + 30_000,
        },
        {
            "case_id": "test-purge", "source_index": 2, "service": "webservice1",
            "labelled_service": "webservice1", "fault_type": "cpu_anomalies",
            "start_ms": blocks[1].start_ms + 100_000,
            "end_ms": blocks[1].start_ms + 130_000,
        },
    ])


class V3RcaCaseRegistryTests(unittest.TestCase):
    def test_detected_anchor_and_w300_boundary_purge(self):
        config = _v3_config()
        registry = _registry(config)
        test_keep = registry.loc[registry.case_id == "test-keep"].iloc[0]
        test_purge = registry.loc[registry.case_id == "test-purge"].iloc[0]
        matching = pd.DataFrame([
            {
                "split": "test", "prediction_id": "pred-keep", "case_id": "test-keep",
                "t_hat": int(test_keep.start_ms + 60_000),
                "gt_start_ms": int(test_keep.start_ms),
                "gt_service": "dbservice2", "fault_type": "memory_anomalies",
                "match_status": "matched",
            },
            {
                "split": "test", "prediction_id": "pred-purge", "case_id": "test-purge",
                "t_hat": int(test_purge.start_ms),
                "gt_start_ms": int(test_purge.start_ms),
                "gt_service": "webservice1", "fault_type": "cpu_anomalies",
                "match_status": "matched",
            },
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matching_path = root / "matching.csv"
            matching.to_csv(matching_path, index=False)
            output = root / "detected_cases.csv"

            # Substitute the loader at the protocol seam so this test stays
            # bounded and does not scan the GAIA raw dataset.
            from unittest import mock
            with mock.patch(
                "scripts.p5.run_i1_rca_features.load_registry",
                return_value=registry,
            ):
                metadata = build_case_registry(
                    config,
                    output,
                    anchor_mode="detected",
                    matching_path=matching_path,
                    config_path=CONFIG_PATH,
                )

            retained = pd.read_csv(output)
            purged = pd.read_csv(root / "detected_cases_purged.csv")
            keep = retained.loc[retained.case_id == "test-keep"].iloc[0]
            self.assertEqual(int(keep.start_ms), int(test_keep.start_ms + 60_000))
            self.assertEqual(int(keep.gt_start_ms), int(test_keep.start_ms))
            self.assertEqual(keep.anchor_type, "detected prediction_available_time")
            self.assertEqual(keep.prediction_id, "pred-keep")
            self.assertIn("test-purge", set(purged.case_id))
            self.assertEqual(metadata["rca_context_purged"], 1)
            self.assertEqual(metadata["split_crossing_purged"], 0)
            self.assertEqual(set(retained.split), {"train", "test"})
            self.assertFalse(any(retained.split.astype(str) == "validation"))

    def test_v3_feature_materialization_rejects_validation_cases(self):
        config = _v3_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_path = root / "cases.csv"
            pd.DataFrame([
                {"case_id": "bad-validation", "start_ms": 1_625_000_000_000, "split": "validation"},
            ]).to_csv(case_path, index=False)
            with self.assertRaisesRegex(ValueError, "Validation"):
                materialize(
                    config,
                    root / "missing-index",
                    root / "features",
                    root / "artifacts",
                    case_path,
                    limit_cases=1,
                    config_path=CONFIG_PATH,
                    formal_result=False,
                )


class V3RcaTrainingTests(unittest.TestCase):
    def test_fit_is_train_only_and_detected_output_keeps_anchor_provenance(self):
        config = _v3_config()
        train_ids = ["train-{}".format(index) for index in range(4)]
        oracle_ids = train_ids + ["oracle-test"]
        detected_ids = train_ids + ["detected-test"]
        train_services = ["dbservice1", "dbservice2", "webservice1", "webservice2"]
        train_faults = ["login_failure", "memory_anomalies", "cpu_anomalies", "file_moving"]
        blocks = temporal_blocks(config)
        train_anchor = blocks[0].start_ms + 600_000
        test_anchor = blocks[1].start_ms + 600_000

        def write_bundle(root, case_ids, test_id, test_service, test_anchor_ms, *, detected=False):
            root.mkdir(parents=True)
            rng = np.random.default_rng(20260913 + int(detected))
            values = rng.normal(0.0, 0.05, size=(len(case_ids), len(GAIA_SERVICES), 68))
            services = train_services + [test_service]
            for row_index, service in enumerate(services):
                values[row_index, GAIA_SERVICES.index(service), 0] += 2.0
            np.save(root / "z2_features.npy", values.astype(np.float32))
            np.save(root / "case_ids.npy", np.asarray(case_ids, dtype="U32"))
            np.save(root / "splits.npy", np.asarray(["train"] * 4 + ["test"], dtype="U10"))
            np.save(
                root / "anchors_ms.npy",
                np.asarray([train_anchor] * 4 + [test_anchor_ms], dtype=np.int64),
            )
            rows = [
                {
                    "case_id": case_id,
                    "service": service,
                    "fault_type": fault,
                    "split": "train",
                    "start_ms": train_anchor,
                }
                for case_id, service, fault in zip(train_ids, train_services, train_faults)
            ]
            rows.append({
                "case_id": test_id,
                "service": test_service,
                "fault_type": "access_permission_denied",
                "split": "test",
                "start_ms": test_anchor_ms,
                **({
                    "gt_start_ms": test_anchor,
                    "anchor_type": "detected prediction_available_time",
                    "prediction_id": "pred-detected",
                    "detection_delay_seconds": 60.0,
                } if detected else {}),
            })
            case_path = root.parent / (root.name + "_cases.csv")
            pd.DataFrame(rows).to_csv(case_path, index=False)
            return case_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle_root = root / "oracle_features"
            detected_root = root / "detected_features"
            oracle_case_path = write_bundle(
                oracle_root, oracle_ids, "oracle-test", "dbservice1", test_anchor
            )
            detected_case_path = write_bundle(
                detected_root, detected_ids, "detected-test", "dbservice1",
                test_anchor + 60_000, detected=True,
            )
            artifact_root = root / "artifacts"
            manifest = train_v3(
                config,
                oracle_root,
                oracle_case_path,
                root / "model.npz",
                artifact_root,
                detected_feature_root=detected_root,
                detected_case_path=detected_case_path,
                config_path=CONFIG_PATH,
                formal_result=False,
            )

            with np.load(root / "model.npz", allow_pickle=False) as arrays:
                self.assertEqual(
                    tuple(np.asarray(arrays["train_case_indices"], dtype=np.int64)),
                    (0, 1, 2, 3),
                )
            self.assertEqual(manifest["training"]["split"], "train only")
            self.assertEqual(manifest["training"]["validation"], "not present in V3")
            detected = pd.read_csv(artifact_root / "rca_detected_predictions.csv")
            self.assertEqual(len(detected), 1)
            self.assertEqual(int(detected.iloc[0].t_hat), test_anchor + 60_000)
            self.assertEqual(int(detected.iloc[0].gt_start_ms), test_anchor)
            self.assertEqual(
                detected.iloc[0].anchor_type,
                "detected prediction_available_time",
            )
            self.assertEqual(detected.iloc[0].prediction_id, "pred-detected")
            self.assertTrue((artifact_root / "root_frequency_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
