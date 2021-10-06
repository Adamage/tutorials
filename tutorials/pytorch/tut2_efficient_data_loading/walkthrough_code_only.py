# # Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import time

import poptorch
import torch
import torch.nn as nn

# Predefine variables that will be useful later
device_iterations = 50
bs = 16
replicas = 1
num_workers = 8


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(41772, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        if self.training:
            return x, self.loss(x, labels)
        return x


if __name__ == '__main__':
    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replicas)

    model = ClassificationModel()
    training_model = poptorch.trainingModel(
        model,
        opts,
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    )

    # Create a fake dataset from random data
    features = torch.randn([10000, 1, 128, 128])
    labels = torch.empty([10000], dtype=torch.long).random_(10)
    dataset = torch.utils.data.TensorDataset(features, labels)

if __name__ == '__main__':
    training_data = poptorch.DataLoader(
        opts,
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        mode=poptorch.DataLoaderMode.Async
    )

if __name__ == '__main__':
    steps = len(training_data)

    t0 = time.time()
    for i, (data, labels) in enumerate(training_data):
        a, b = data, labels
    t1 = time.time()
    total_time = t1 - t0

    print(f"Total execution time: {total_time:.2f} s")

    items_per_second = (steps * device_iterations * bs * replicas) / total_time
    print(f"DataLoader throughput: {items_per_second:.2f} items/s")

if __name__ == '__main__':
    opts.enableSyntheticData(True)
    training_model = poptorch.trainingModel(
        model,
        opts,
        poptorch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    )
    training_data = poptorch.DataLoader(
        opts,
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        mode=poptorch.DataLoaderMode.Async,
        async_options={"early_preload": True}
    )
    steps = len(training_data)

    print("Compiling + Warmup ...")
    data_batch, labels_batch = next(iter(training_data))
    training_model.compile(data_batch, labels_batch)

if __name__ == '__main__':
    print(f"Evaluating: {steps} steps of {device_iterations * bs * replicas} items")

    # With synthetic data enabled, no data is copied from the host to the IPU,
    # so we don't use the dataloader, to prevent influencing the execution
    # time and therefore the IPU throughput calculation
    t0 = time.time()
    for _ in range(steps):
        training_model(data_batch, labels_batch)
    t1 = time.time()
    total_time = t1 - t0

    items_per_second = (steps * device_iterations * bs * replicas) / total_time
    print(f"Total execution time: {total_time:.2f} s")
    print(f"IPU throughput: {items_per_second:.2f} items/s")

def validate_model_performance(dataset, device_iterations=50,
                               batch_size=16, replicas=4, num_workers=8,
                               synthetic_data=False):
    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replicas)
    if synthetic_data:
        opts.enableSyntheticData(True)

    training_data = poptorch.DataLoader(opts, dataset=dataset, batch_size=batch_size,
                                        shuffle=True, drop_last=True,
                                        num_workers=num_workers,
                                        mode=poptorch.DataLoaderMode.Async,
                                        async_options={"early_preload": True})
    steps = len(training_data)

    t0 = time.time()
    for data_batch, labels_batch in training_data:
        pass
    t1 = time.time()
    total_time = t1 - t0
    items_per_second = (steps * device_iterations * bs * replicas) / total_time

    print(f"DataLoader throughput: {items_per_second:.2f} items/s")

    if synthetic_data:
        # With synthetic data enabled, no data is copied from the host to the
        # IPU, so we don't use the dataloader, to prevent influencing the
        # execution time and therefore the IPU throughput calculation
        t0 = time.time()
        for _ in range(steps):
            training_model(data_batch, labels_batch)
        t1 = time.time()
    else:
        t0 = time.time()
        for data, labels in training_data:
            training_model(data, labels)
        t1 = time.time()

    total_time = t1 - t0
    items_per_second = (steps * device_iterations * bs * replicas) / total_time
    print(f"IPU throughput: {items_per_second:.2f} items/s")

if __name__ == '__main__':
    print('*** EXPERIMENTS ***')

    print('Global batch size 16 with synthetic data')
    validate_model_performance(dataset, batch_size=16, replicas=1,
                               device_iterations=50, num_workers=8,
                               synthetic_data=True)

    print("\nGlobal batch size 16 with real data:")
    validate_model_performance(dataset, batch_size=16, replicas=1,
                               device_iterations=50, num_workers=8,
                               synthetic_data=False)


if __name__ == '__main__':
    print('\nGlobal batch size 64 with synthetic data:')
    validate_model_performance(dataset, batch_size=16, replicas=4,
                               device_iterations=50, num_workers=8,
                               synthetic_data=True)

    print("\nGlobal batch size 64 with real data:")
    validate_model_performance(dataset, batch_size=16, replicas=4,
                               device_iterations=50, num_workers=8,
                               synthetic_data=False)
