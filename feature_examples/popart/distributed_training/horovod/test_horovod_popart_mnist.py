# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import pytest
import subprocess

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker, \
    parse_results_with_regex, verify_model_accuracies


class TestHorovodPopARTMNISTImageClassification(SubProcessChecker):
    """High-level integration tests for training with the MNIST data-set"""

    def locate_horovodrun():
        whereis_horovodrun = subprocess.run(["whereis", "horovodrun"], stdout=subprocess.PIPE, universal_newlines=True).stdout.splitlines()
        path_to_horovodrun = whereis_horovodrun[0].split(": ")[1]

        return path_to_horovodrun

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.path.dirname(__file__)
        cls.accuracy_tolerances = 3.0
        cls.path_to_horovodrun = cls.locate_horovodrun()

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_two_processes(self):
        """Generic test on default arguments in training"""

        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --epochs 3"
        output = self.run_command(cmd, self.cwd, "Accuracy")
        expected_accuracy = [71.23, 78.75, 82.11]
        accuracies = parse_results_with_regex(output, r".* + Accuracy=+([\d.]+)%")
        verify_model_accuracies(accuracies[0], expected_accuracy, self.accuracy_tolerances)

    # @pytest.mark.ipus(16)
    # @pytest.mark.category2
    # def test_mnist_train_sixteen_processes(self):
    #     """Generic test on default arguments in training"""
    #     cmd = f"python3 {self.path_to_horovodrun} -np 16 -H localhost:16 python horovod_popart_mnist.py --epochs 3"
    #     output = self.run_command(cmd, self.cwd, "Accuracy")
    #     expected_accuracy = [83.75, 83.86, 84.79]
    #     accuracies = parse_results_with_regex(output, r".* + Accuracy=+([\d.]+)%")
    #     verify_model_accuracies(accuracies[0], expected_accuracy, self.accuracy_tolerances)



    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_mnist_train_pipelined_processes(self):
        """Generic test on default arguments in training"""
        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --num-ipus 2 --pipeline --epochs 3"
        output = self.run_command(cmd, self.cwd, "Accuracy")
        expected_accuracy = [71.26, 78.77, 82.14]
        accuracies = parse_results_with_regex(output, r".* + Accuracy=+([\d.]+)%")
        verify_model_accuracies(accuracies[0], expected_accuracy, self.accuracy_tolerances)



    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_mnist_train_synthetic_data(self):
        """Generic test on default arguments in training"""
        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --epochs 1 --syn-data-type random_normal"
        output = self.run_command(cmd, self.cwd, "Accuracy")


    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_log_graph_trace(self):
        """Generic test on default arguments in training"""
        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --epochs 1 --log-graph-trace"
        output = self.run_command(cmd, self.cwd, "Accuracy")

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_test_mode(self):
        """Generic test on default arguments in training"""
        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --epochs 1 --test-mode inference"
        output = self.run_command(cmd, self.cwd, "Accuracy")


    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_simulation(self):
        """Generic test on default arguments in training"""
        cmd = f"python3 {self.path_to_horovodrun} -np 2 -H localhost:2 python3 horovod_popart_mnist.py --epochs 1 --simulation"
        output = self.run_command(cmd, self.cwd, "Accuracy")


    # @pytest.mark.ipus(8)
    # @pytest.mark.category2
    # def test_mnist_train_multiple_options(self):
    #     """Generic test on default arguments in training"""
    #     cmd = f"python3 {self.path_to_horovodrun} -np 4 -H localhost:4 python horovod_popart_mnist.py --epochs 8 --batch-size 64 --batches-per-step 50 --num-ipus 2 --pipeline --log-graph-trace"
    #     output = self.run_command(cmd, self.cwd, "Accuracy")
    #     expected_accuracy = [69.80, 77.77, 81.21, 83.01, 84.11, 84.76, 85.48, 86.02]
    #     accuracies = parse_results_with_regex(output, r".* + Accuracy=+([\d.]+)%")
    #     verify_model_accuracies(accuracies[0], expected_accuracy, self.accuracy_tolerances)
