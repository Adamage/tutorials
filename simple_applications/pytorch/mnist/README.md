# PyTorch(PopTorch) MNIST Training Demo

This example demonstrates how to train a network on the MNIST dataset using
PopTorch. To learn more about PopTorch, see our [PyTorch for the IPU: User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html).

## How to use this demo

### Environment preparation

Install the Poplar SDK following the instructions in the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html)
guide for your IPU system. Make sure to run the `enable.sh` scripts for Poplar 
and PopART and activate a Python3 virtualenv with PopTorch installed.

Then install the package requirements:
```bash
pip install -r requirements.txt
```

### Setting hyperparameters
Set the hyperparameters for this demo. If you're running this example in 
a Jupyter notebook and wish to modify them, re-run all the cells below.


```python
# Learning rate.
learning_rate = 0.03

# Number of epochs to train.
epochs = 10

# Batch size for training.
batch_size = 8

# Batch size for testing.
test_batch_size = 80

# Device iteration - batches per step. Number of iterations the device should
# run over the data before returning to the user.
# This is equivalent to running the IPU in a loop over that the specified
# number of iterations, with a new batch of data each time. However, increasing
# deviceIterations is more efficient because the loop runs on the IPU directly.
device_iterations = 50
```

## Training a PopTorch model for MNIST classification

Import required libraries:


```python
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch
import torch.optim as optim
```

Download the datasets for MNIST and set up data loaders.
Source: [The MNIST Database](http://yann.lecun.com/exdb/mnist/).


```python
local_dataset_path = '~/.torch/datasets'

transform_mnist = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ]
)

training_dataset = torchvision.datasets.MNIST(
        local_dataset_path,
        train=True,
        download=True,
        transform=transform_mnist
)

training_data = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size * device_iterations,
    shuffle=True,
    drop_last=True
)

test_dataset = torchvision.datasets.MNIST(
        local_dataset_path,
        train=False,
        download=True,
        transform=transform_mnist
)

test_data = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    drop_last=True
)
```

Let's define the elements of our neural network. We first create a `Block`
instance consisting of a 2D convolutional layer with pooling, followed by
a ReLU activation.


```python
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
```

Now, let's construct our neural network.


```python
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 32, 3, 2)
        self.layer2 = Block(32, 64, 3, 2)
        self.layer3 = nn.Linear(1600, 128)
        self.layer3_act = nn.ReLU()
        self.layer3_dropout = torch.nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Flatten layer
        x = x.view(-1, 1600)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        return x
```

Next we define a thin wrapper around the `torch.nn.Module` that will use
the cross-entropy loss function.

This class is creating a custom module to compose the Neural Network and 
the Cross Entropy module into one object, which under the hood will invoke 
the `__call__` function on `nn.Module` and consequently the `forward` method.


```python
class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        loss = self.loss(output, loss_inputs)
        return output, loss
```

Let's initialise the neural network from our defined classes.


```python
model = Network()
model_with_loss = TrainingModelWithLoss(model)
model_opts = poptorch.Options().deviceIterations(device_iterations)
```

Next we will set the `AnchorMode` for our training. By default, PopTorch will
return to the host machine only a limited set of information for performance
reasons. This is represented by having `AnchorMode.Final` as the default, which
means that only the final batch of the internal loop is returned to the host.
When inspecting the training performance as it is executing, values like 
accuracy or losses will then be calculated only for that last batch, 
specifically the `batch_size` out of the whole step which is 
`batch_size*device_iterations`.
We can set this to `AnchorMode.All` to be able to present the full information.
This has an impact on the speed of training, due to overhead of transferring
more data to the host machine.
To learn about all values for `AnchorMode`, please see the [PopTorch API documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html?highlight=anchorMode#poptorch.Options.anchorMode).


```python
model_opts = model_opts.anchorMode(poptorch.AnchorMode.All)
```

We can check if the model is assembled correctly by printing the string 
representation of the model object.


```python
print(model_with_loss)
```

    TrainingModelWithLoss(
      (model): Network(
        (layer1): Block(
          (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
          (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (relu): ReLU()
        )
        (layer2): Block(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
          (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (relu): ReLU()
        )
        (layer3): Linear(in_features=1600, out_features=128, bias=True)
        (layer3_act): ReLU()
        (layer3_dropout): Dropout(p=0.5, inplace=False)
        (layer4): Linear(in_features=128, out_features=10, bias=True)
      )
      (loss): CrossEntropyLoss()
    )


Now we apply the model wrapping function, which will perform a shallow copy
of the PyTorch model. To train the model, we will use [SGD](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.optim.SGD),
the Stochastic Gradient Descent with no momentum.


```python
training_model = poptorch.trainingModel(
    model_with_loss,
    model_opts,
    optimizer=optim.SGD(model.parameters(), lr=learning_rate)
)
```

We are ready to start training. However to track the accuracy while training
we need to define one more helper function. During the training, not every 
samples prediction is returned for efficiency reasons, so this helper function
will check accuracy for labels where prediction is available. This behavior
is controlled by setting `AnchorMode` in `poptorch.Options()`.


```python
def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0]:]
    accuracy = \
        torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy
```

This code will perform the training over the requested amount of epochs
and batches using the configured Graphcore IPUs.


```python
nr_batches = len(training_data)

for epoch in tqdm(range(1, epochs+1), leave=True, desc="Epochs", total=epochs):
    with tqdm(training_data, total=nr_batches, leave=False) as bar:
        for data, labels in bar:
            preds, losses = training_model(data, labels)

            mean_loss = torch.mean(losses).item()

            acc = accuracy(preds, labels)
            bar.set_description(
                "Loss: {:0.4f} | Accuracy: {:05.2F}% ".format(mean_loss, acc)
            )
```

Release resources:


```python
training_model.detachFromDevice()
```

## Evaluating the trained model

Let's check the validation loss on IPU using the trained model. The weights 
in `model.parameters()` will be copied from the IPU to the host. The weights
from the trained model will be reused to compile the new inference model.


```python
inference_model = poptorch.inferenceModel(model)
```

Perform validation.


```python
nr_batches = len(test_data)
sum_acc = 0.0
with tqdm(test_data, total=nr_batches, leave=False) as bar:
    for data, labels in bar:
        output = inference_model(data)
        sum_acc += accuracy(output, labels)
```

Finally the accuracy on the test set is:


```python
print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))
```

    Accuracy on test set: 99.29%


Release resources:


```python
inference_model.detachFromDevice()
```
