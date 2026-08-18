from pathlib import Path
import tempfile
import unittest

from src.data.gaia import _repair_float_cpu_duration
from src.data.rcaeval import load_re2ob_cases


class GAIAAdapterTest(unittest.TestCase):
    def test_float_cpu_duration_is_repaired(self):
        parsed = {
            "anomaly_type": "[cpu_anomalies]",
            "st_time": "2021-07-27 23:46:25.446893+08:00",
            "ed_time": "",
            "duration": 0,
            "message": "[cpu_anomalies] starts now and lasts 3.0210864543914795 seconds",
        }
        _repair_float_cpu_duration(parsed)
        self.assertAlmostEqual(parsed["duration"], 3.0210864543914795)
        self.assertTrue(str(parsed["ed_time"]).startswith("2021-07-27 23:46:28"))


class RCAEvalAdapterTest(unittest.TestCase):
    def make_case(self, root: Path, condition="emailservice_mem", replicate="1"):
        case_directory = root / condition / replicate
        case_directory.mkdir(parents=True)
        (case_directory / "inject_time.txt").write_text("1705475505", encoding="utf-8")
        (case_directory / "metrics.csv").write_text(
            "time,emailservice_x\n1,0\n", encoding="utf-8"
        )
        (case_directory / "simple_metrics.csv").write_text(
            "time,emailservice_cpu,emailservice_mem,frontend_cpu,frontend_mem,istio-init_cpu\n"
            "1,0,0,0,0,0\n",
            encoding="utf-8",
        )
        (case_directory / "logs.csv").write_text(
            "timestamp,container_name\n1,emailservice\n", encoding="utf-8"
        )
        (case_directory / "traces.csv").write_text(
            "startTime,serviceName\n1,emailservice\n", encoding="utf-8"
        )
        return case_directory

    def test_case_paths_are_hidden_behind_opaque_input_uris(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            case_directory = self.make_case(raw_root)
            result = load_re2ob_cases(str(raw_root))

            self.assertEqual(len(result.inputs), 1)
            self.assertEqual(len(result.labels), 1)
            self.assertEqual(result.inputs[0].services, ("emailservice", "frontend"))
            self.assertEqual(result.labels[0].root_service, "emailservice")
            self.assertNotIn("emailservice_mem", result.inputs[0].case_id)
            self.assertNotIn("emailservice_mem", result.inputs[0].metrics.uri)
            self.assertIn("emailservice_mem", result.sources[0].relative_directory)
            self.assertEqual(
                result.sources[0].metrics_path, str(case_directory / "metrics.csv")
            )

    def test_missing_modality_is_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            case_directory = self.make_case(raw_root)
            (case_directory / "traces.csv").unlink()
            result = load_re2ob_cases(str(raw_root))
            self.assertEqual(result.inputs, ())
            self.assertEqual(len(result.excluded), 1)
            self.assertIn("traces", result.excluded[0].reason)


if __name__ == "__main__":
    unittest.main()
