# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch
import torch.optim as optim

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

class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        loss = self.loss(output, loss_inputs)
        return output, loss

model = Network()
model_with_loss = TrainingModelWithLoss(model)
model_opts = poptorch.Options().deviceIterations(device_iterations)

model_opts = model_opts.anchorMode(poptorch.AnchorMode.All)

print(model_with_loss)

training_model = poptorch.trainingModel(
    model_with_loss,
    model_opts,
    optimizer=optim.SGD(model.parameters(), lr=learning_rate)
)

def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0]:]
    accuracy = \
        torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy

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

training_model.detachFromDevice()

inference_model = poptorch.inferenceModel(model)

nr_batches = len(test_data)
sum_acc = 0.0
with tqdm(test_data, total=nr_batches, leave=False) as bar:
    for data, labels in bar:
        output = inference_model(data)
        sum_acc += accuracy(output, labels)

print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))

inference_model.detachFromDevice()
