import io
import unittest
from unittest.mock import patch

from util.train import _batch_progress_enabled


class TrainingLoggingTest(unittest.TestCase):
    def test_non_tty_defaults_to_quiet_batch_progress(self):
        self.assertFalse(_batch_progress_enabled(stream=io.StringIO()))

    def test_explicit_environment_override(self):
        with patch.dict("os.environ", {"P5_AD_BATCH_PROGRESS": "1"}, clear=False):
            self.assertTrue(_batch_progress_enabled())
        with patch.dict("os.environ", {"P5_AD_BATCH_PROGRESS": "0"}, clear=False):
            self.assertFalse(_batch_progress_enabled())

    def test_explicit_argument_wins(self):
        with patch.dict("os.environ", {"P5_AD_BATCH_PROGRESS": "0"}, clear=False):
            self.assertTrue(_batch_progress_enabled(True))
            self.assertFalse(_batch_progress_enabled(False))


if __name__ == "__main__":
    unittest.main()
