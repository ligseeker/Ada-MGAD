import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.p5.run_i1_e2e import evaluate
from src.e2e.protocol import GAIA_SERVICES, load_config, sha256_file
from src.e2e.rca_model import fit_conditional_logit, save_conditional_logit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmallEndToEndIntegrationTest(unittest.TestCase):
    def test_detected_anchor_evaluation_writes_all_formal_outputs(self):
        config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts"
            artifact.mkdir()
            index_root = root / "index"
            index_root.mkdir()
            anchor = int(config["split"]["boundaries_ms"][1]) + 600_000
            timestamps = np.arange(anchor - 300_000, anchor + 300_000, 15_000, dtype=np.int64)
            values = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)
            np.save(index_root / "metric.timestamps.npy", timestamps, allow_pickle=False)
            np.save(index_root / "metric.values.npy", values, allow_pickle=False)
            index_manifest = index_root / "index_manifest.json"
            index_manifest.write_text(json.dumps({
                "metric_series": [{
                    "service": "dbservice1", "indicator": "cpu",
                    "timestamps": "metric.timestamps.npy", "values": "metric.values.npy",
                    "timestamps_sha256": sha256_file(index_root / "metric.timestamps.npy"),
                    "values_sha256": sha256_file(index_root / "metric.values.npy"),
                }],
                "logs": {},
                "traces": {"parts": []},
            }), encoding="utf-8")

            rng = np.random.default_rng(3)
            training = rng.normal(size=(3, 10, 68))
            model = fit_conditional_logit(training, [0, 1, 2], train_indices=[0, 1, 2])
            model_path = root / "model.npz"
            save_conditional_logit(model_path, model)

            matching_path = root / "event_matching.csv"
            pd.DataFrame([{
                "split": "test", "prediction_id": "test-pred-000000", "t_hat": anchor,
                "match_status": "matched", "case_id": "case-0", "gt_service": "dbservice1",
                "fault_type": "login failure", "gt_start_ms": anchor - 10_000,
                "detection_delay_seconds": 10.0,
            }]).to_csv(matching_path, index=False)
            node_path = root / "ad_test_predictions.csv"
            pd.DataFrame([{
                "split": "test", "prediction_timestamp": anchor, "service": service,
                "anomaly_score": 1.0 if index == 0 else 0.0,
            } for index, service in enumerate(GAIA_SERVICES)]).to_csv(node_path, index=False)
            prediction = {
                "case_id": "case-0",
                "ranking_json": json.dumps(GAIA_SERVICES),
                **{"rank_{}".format(index + 1): service for index, service in enumerate(GAIA_SERVICES)},
            }
            oracle_path = root / "rca_oracle_predictions.csv"
            frequency_path = root / "root_frequency_predictions.csv"
            pd.DataFrame([prediction]).to_csv(oracle_path, index=False)
            pd.DataFrame([prediction]).to_csv(frequency_path, index=False)
            case_registry_path = root / "rca_case_registry.csv"
            pd.DataFrame([{
                "case_id": "case-0", "split": "test", "service": "dbservice1",
                "fault_type": "login failure", "start_ms": anchor - 10_000,
            }]).to_csv(case_registry_path, index=False)
            (artifact / "rca_metrics.json").write_text(
                json.dumps({"schema_version": "fixture"}), encoding="utf-8"
            )

            result = evaluate(
                config, artifact, index_manifest, model_path,
                root / "detected.npy", matching_path, node_path, oracle_path,
                frequency_path, case_registry_path,
            )
            self.assertEqual(
                result["e2e_diagnosis_metrics"]["counts"]["matched_events"], 1
            )
            self.assertTrue((artifact / "rca_detected_predictions.csv").is_file())
            self.assertTrue((artifact / "detector_only_predictions.csv").is_file())
            self.assertTrue((artifact / "e2e_diagnosis_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
