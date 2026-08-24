import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import main


class FileLoggingTest(unittest.TestCase):
    def test_stdout_is_mirrored_to_default_train_log(self):
        with tempfile.TemporaryDirectory() as directory:
            state = main.start_file_logging(SimpleNamespace(
                output_dir=directory,
                log_file=str(Path(directory, 'train.log')),
                no_file_log=False,
            ))
            try:
                print('file-log-test-marker')
            finally:
                main.stop_file_logging(state)

            content = Path(directory, 'train.log').read_text(encoding='utf-8')
            self.assertIn('file-log-test-marker', content)
            self.assertIn('experiment start', content)
            self.assertIn('experiment end', content)


if __name__ == '__main__':
    unittest.main()
