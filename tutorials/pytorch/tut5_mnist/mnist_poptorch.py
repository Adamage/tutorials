#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
# PyTorch(PopTorch) MNIST Training Demo

This example demonstrates how to train a network on the MNIST dataset using
PopTorch.
"""
"""
## How to use this demo

1) Prepare the environment.

Install the Poplar SDK following the instructions in the Getting Started guide 
for your IPU system. Make sure to run the `enable.sh` scripts for Poplar and 
PopART and activate a Python virtualenv with PopTorch installed.

Then install the package requirements:
```bash
pip install -r requirements.txt
```

2) Run the program. Note that the PopTorch Python API only supports Python 3.
Data will be automatically downloaded using torch vision utils.

```bash
python3 mnist_poptorch.py
```
"""
"""
Select your hyperparameters in this cell. If you wish to modify them, re-run
all cells below it. Further reading [Hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
Setup parameters for training:
"""
# Batch size for training
batch_size = 8

# Device iteration - batches per step
batches_per_step = 50

# Batch size for testing
test_batch_size = 80

# Number of epochs to train
epochs = 10

# Learning rate
lr = 0.05
"""
Import required libraries:
"""
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch
import torch.optim as optim
"""
Download the datasets for MNIST - database for handwritten digits.
Source: [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
"""

# The following is a workaround for pytorch issue #1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

local_dataset_path = 'mnist_data/'

training_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        local_dataset_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))]
        )
    ),
    batch_size=batch_size * batches_per_step,
    shuffle=True,
    drop_last=True
)

test_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        local_dataset_path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))]
        )
    ),
    batch_size=test_batch_size,
    shuffle=True,
    drop_last=True
)
# sst_hide_output
"""
Let's define the elements of our neural network. First, we create the class
`Block` which while define a simple 2D convolutional layer with pooling and
a rectified linear unit (ReLU). To see explanation of pooling and ReLU, see:
[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network#Building_blocks)
"""
class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

"""
Now let's construct the deep neural network with 4 Convolutional layers and
a [softmax layer](https://en.wikipedia.org/wiki/Softmax_function) at the output.
"""
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 32, 3, 2)
        self.layer2 = Block(32, 64, 3, 2)
        self.layer3 = nn.Linear(1600, 128)
        self.layer3_act = nn.ReLU()
        self.layer3_dropout = torch.nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Flatten layer
        x = x.view(-1, 1600)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        x = self.softmax(x)
        return x

"""
Here we define a thin wrapper around the `torch.nn.Module` that will use
cross-entropy loss function - see more [here](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)
"""
class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        if loss_inputs is None:
            return output
        else:
            loss = self.loss(output, loss_inputs)
            return output, loss
"""
Let's initiate the neural network from our defined classes.
"""
model = Network()
model_with_loss = TrainingModelWithLoss(model)
model_opts = poptorch.Options().deviceIterations(batches_per_step)
"""
Now we apply the model wrapping function, which will perform a shallow copy
of the PyTorch model. To perform the machine learning operations, we also
will use the Stochastic Gradient Descent with no momentum [SGD](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.optim.SGD).
"""
training_model = poptorch.trainingModel(
    model_with_loss,
    model_opts,
    optimizer=optim.SGD(model.parameters(), lr=lr)
)
"""
We are ready to start training. However to track the accuracy while training
we need to define one more helper function. During the training, not every 
samples prediction is returned for efficiency reasons, so this helper function
will check accuracy for labels where prediction is available.
"""
def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0]:]
    accuracy = \
        torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy
"""
This code will perform the requested amount of epochs and batches using the
configured Graphcore IPUs.
"""
nr_batches = len(training_data)
for epoch in range(1, epochs+1):
    print("Epoch {0}/{1}".format(epoch, epochs))
    with tqdm(training_data, total=nr_batches, leave=False) as bar:
        for data, labels in bar:
            preds, losses = training_model(data, labels)
            with torch.no_grad():
                mean_loss = torch.mean(losses).item()
                acc = accuracy(preds, labels)
            bar.set_description(
                "Loss:{:0.4f} | Accuracy:{:0.2f}%".format(mean_loss, acc)
            )
# sst_hide_output
"""
Update the weights in model by copying from the training IPU. 
This updates `model.parameters()`.
"""
training_model.copyWeightsToHost()

"""
Release resources:
"""
training_model.detachFromDevice()
"""
Check validation loss on IPU once trained. Because PopTorch will be compiled 
on first call the weights in `model.parameters()` will be copied implicitly. 
Subsequent calls will need to call `inference_model.copyWeightsToDevice()`.
"""
inference_model = poptorch.inferenceModel(model)

"""
Perform validation
"""
nr_batches = len(test_data)
sum_acc = 0.0
with torch.no_grad():
    with tqdm(test_data, total=nr_batches, leave=False) as bar:
        for data, labels in bar:
            output = inference_model(data)
            sum_acc += accuracy(output, labels)
# sst_hide_output
"""
Finally the accuracy on the test set is:
"""
print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))
"""
Release resources:
"""
inference_model.detachFromDevice()
