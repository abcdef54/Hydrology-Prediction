"""Device selection without requiring a physical GPU."""

import unittest
from unittest.mock import patch

from src.backend.prediction.device import resolve_prediction_device


class PredictionDeviceTests(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_uses_cpu_without_cuda(self, available):
        self.assertEqual(resolve_prediction_device(), "cpu")

    @patch("torch.cuda.get_device_name", return_value="Test GPU")
    @patch("torch.cuda.synchronize")
    @patch("torch.mm")
    @patch("torch.ones")
    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_checks_cuda_execution(self, available, ones, mm, synchronize, name):
        self.assertEqual(resolve_prediction_device("auto"), "cuda")
        ones.assert_called_once_with((2, 2), device="cuda")
        mm.assert_called_once_with(ones.return_value, ones.return_value)
        synchronize.assert_called_once_with()

    @patch("torch.ones", side_effect=RuntimeError("incompatible CUDA driver"))
    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_falls_back_when_cuda_execution_fails(self, available, ones):
        with self.assertLogs("src.backend.prediction.device", level="WARNING"):
            self.assertEqual(resolve_prediction_device(), "cpu")

    @patch("torch.cuda.is_available")
    def test_explicit_device_is_preserved(self, available):
        self.assertEqual(resolve_prediction_device("cpu"), "cpu")
        self.assertEqual(resolve_prediction_device("cuda:1"), "cuda:1")
        available.assert_not_called()
