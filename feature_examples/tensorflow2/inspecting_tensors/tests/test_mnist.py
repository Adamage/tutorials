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
    def test_script_execution(self):
        """ Script will run with the default values for all variables.
        """
        self.run_command("python3 mnist.py",
                         working_path,
                         [])
