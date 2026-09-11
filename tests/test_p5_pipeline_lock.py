import tempfile
import unittest
from pathlib import Path

from scripts.p5.run_i1_pipeline import exclusive_pipeline_lock


class PipelineLockTests(unittest.TestCase):
    def test_second_writer_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "pipeline.lock"
            with exclusive_pipeline_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "owns the shared output lock"):
                    with exclusive_pipeline_lock(lock_path):
                        pass
            with exclusive_pipeline_lock(lock_path):
                pass


if __name__ == "__main__":
    unittest.main()
