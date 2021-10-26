# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


from pathlib import Path
import pytest
import tensorflow as tf
import unittest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).resolve().parent.parent


@pytest.mark.category2
@unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
class TestTensorFlow2InspectingTensors(SubProcessChecker):
    """Integration tests for TensorFlow 2 Inspecting Tensors example"""


    @pytest.mark.ipus(2)
    def test_default_commandline(self):
        """ Test the default command line
        """
        self.run_command("python3 mnist.py",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(125, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(500, 32, 128\)\n" +
                          r"key: Dense_10_acts shape: \(500, 32, 10\)"])

    @pytest.mark.ipus(1)
    def test_model_gradient_accumulation_pre_accumulated_gradients(self):
        """ Test the Model, outfeeding the pre-accumulated gradients
        """
        self.run_command("python3 mnist.py"
                         " --epochs 1 --steps-per-epoch 500 "
                         " --outfeed-pre-accumulated-gradients",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(500, 32, 128\)"])

    @pytest.mark.ipus(2)
    def test_pipeline_model(self):
        """ Test the pipelined Model, outfeeding the pre-accumulated gradients
        """
        self.run_command("python3 mnist.py --epochs 1"
                         " --steps-per-epoch 500 "
                         "--outfeed-pre-accumulated-gradients"
                         " --activations-filters Dense_10",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_128\/bias:0_grad shape: \(500, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_10_acts shape: \(500, 32, 10\)"])

    @pytest.mark.ipus(1)
    def test_no_gradient_filters(self):
        """ Test with gradients-filters as 'none'
        """
        self.run_command("python3 mnist.py"
                         " --epochs 1 --steps-per-epoch 500"
                         " --gradients-filters none",
                         working_path,
                         [r"Gradients callback\n" +
                          r"key: Dense_10\/bias:0_grad shape: \(125, 10\)",
                          r"key: Dense_128\/bias:0_grad shape: \(125, 128\)",
                          r"Multi-layer activations callback\n" +
                          r"key: Dense_128_acts shape: \(500, 32, 128\)"])
