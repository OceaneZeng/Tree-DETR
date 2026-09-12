"""Exercise the launcher probe in a subprocess without CUDA dependencies."""

import os
import subprocess
import unittest
from unittest import mock

from tools.owod import run_mowodb_reimplementations as runner


class CudaPreflightTests(unittest.TestCase):
    def run_probe(self, capabilities, available=True):
        stub = (
            "import sys, types\n"
            f"caps = {capabilities!r}\n"
            "sys.modules['torch'] = types.SimpleNamespace(\n"
            "    __version__='2.4.1', cuda=types.SimpleNamespace(\n"
            f"        is_available=lambda: {available!r},\n"
            "        get_arch_list=lambda: ['sm_86', 'sm_90'],\n"
            "        device_count=lambda: len(caps),\n"
            "        get_device_capability=lambda i: caps[i]))\n"
        )
        execute = subprocess.run

        def launch(command, **kwargs):
            self.assertEqual(command[:2], [runner.sys.executable, '-c'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1')
            return execute(command[:2] + [stub + command[2]], **kwargs)

        environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           CUDA_VISIBLE_DEVICES='0,1')
        with mock.patch.object(runner.subprocess, 'run', side_effect=launch):
            runner.validate_cuda_runtime(environment)

    def test_two_3090_devices_pass_without_raising_none(self):
        self.run_probe([(8, 6), (8, 6)])

    def test_unsupported_5090_reports_architecture(self):
        with self.assertRaisesRegex(RuntimeError, 'unsupported GPU architectures.*sm_120'):
            self.run_probe([(8, 6), (12, 0)])

    def test_unavailable_cuda_reports_cause(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
            self.run_probe([], available=False)


if __name__ == '__main__':
    unittest.main()
