"""
Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
"""
# TensorFlow 1 Pipelining Tutorial
"""
"""
## Introduction
If a model is too big to fit on one IPU, you will need to distribute it over 
multiple IPUs. This is called model parallelism.  
Documentation for model parallelism on IPU with TensorFlow can be found here: 
[TensorFlow Model Parallelism](<https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/model.html#model-parallelism>)

This tutorial first provides an overview of the key concepts from this document. 
It then provides a walkthrough of how pipelining can be applied to an existing 
TensorFlow application that currently runs on a single IPU.
"""
"""
## Requirements

For software installation and setup details, please see the Getting Started 
guide for your hardware setup, available here: 
[Getting Started Guides](<https://docs.graphcore.ai/en/latest/getting-started.html>).

You must have installed the Graphcore TensorFlow 1 wheel into your current 
active Python environment before starting the tutorial.
"""
"""
## Directory Structure

| Filename                               | Description                                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `README.md`                            | This tutorial in markdown format                                                               |
| `pipelining.ipynb`                     | This tutorial in Jupyter notebook format                                                       |
| `pipelining.py`                        | An executable python file that is used as a single source to generate this file                |
| `step1_single_ipu.py`                  | Step 1 - the existing TensorFlow application that runs **without** pipelining on a single IPU. |
| `answers/step2_sharding.py`            | Step 2 - shows how to run on multiple IPUs, still **without** pipelining.                      |
| `answers/step3_pipelining.py`          | Step 3 - shows how to add pipelining.                                                          |
| `answers/step4_configurable_stages.py` | Step 4 - shows how configurable stages might be implemented.                                   |
| `scripts/profile.sh`                   | Helper script to capture profiling reports.                                                    |
| `images/`                              | Images used in this README.                                                                    |

"""
"""
## Key Principles of Model Pipelining

This section describes the key principles of model pipelining with TensorFlow 
1 on IPU. A later section will apply this to the existing TensorFlow application 
that initially runs on a single IPU.

"""
"""
### Overview

TensorFlow on IPU supports two methods for model parallelism.
"""
"""
#### 1. Model Parallelism With Sharding

With model sharding, the model is split into stages where each stage can fit 
and be run on a single IPU. The output of each stage is fed to the input of 
the stage that follows it. Execution of the model is serialised. That is, each 
stage is executed in turn while the IPUs allocated to other stages remain idle.

![Sharding outline](images/sharding_outline.png)

Refer to the technical note on TensorFlow Model Parallelism for full details: 
[TensorFlow Model Parallelism - Sharding](<https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/sharding.html#sharding>)

Model sharding provides a method to run larger models that is conceptually 
straightforward and might be useful for initial development or debugging. 
However, it does not offer good utilisation of the allocated IPU resource and, 
for this reason, sharding is not recommended for production models where 
performance is critical.
"""

"""
#### 2. Model Parallelism With Pipelining

With pipelining, as with sharding, the model is split into stages where each
stage can fit and be run on a single IPU. However, unlike sharding, the compute
for separate batches is overlapped so that execution of the model is parallelised.
That is, each stage (part of the original model) is executed on its IPU while
the IPUs allocated to previous stages are already working on subsequent batches.
This provides improved utilisation compared to sharding.
"""
"""
![Pipelining outline](images/pipelining_outline.png)

Refer to the technical note on TensorFlow Model Parallelism for full details:
[TensorFlow Model Parallelism - Pipelining](<https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html#pipelining>)

Pipelining provides a method to run larger models that is conceptually less
straightforward compared to sharding. However, it offers better utilisation of
the allocated IPU resource and, for this reason, pipelining is recommended
where performance is critical.

This tutorial focuses on how to apply pipelining in TensorFlow 1.
"""
"""
### Pipeline Execution Phases

It is important to understand the key phases of pipeline execution:

1. Ramp up -  the pipeline is being filled; work is flowing into each stage 
until all stages are filled (all IPUs are busy).
2. Main execution - all stages are filled and IPU utilisation is maximised.
3. Ramp down - the pipeline is being drained; work is flowing out of each stage 
until all stages are empty (no IPUs are busy).
4. Weight updates - all pipeline batches have been processed, so accumulated 
gradients can be processed (gradient descent) and weights updated.

Note:  

* Each individual batch passed through the pipeline is called a **mini-batch**.  
* Weights are updated only once a set of mini-batches has been fully processed.  
* Gradients are accumulated across a set of mini-batches.  
* Weight updates are applied once all the complete set of mini-batches are processed.  

In short, pipelining enforces **gradient accumulation** where:  

`effective batch size` = `mini-batch size` * `gradient accumulation count`  

Performing gradient accumulation is valid because summing the gradients across 
all the examples in a batch immediately and accumulating them over several steps are equivalent.  

Increasing the gradient accumulation count has these benefits:

1. A smaller proportion of time is spent in the ramp up and ramp down - that is, 
more time is spent in the main execution phase where maximum utilisation of the 
IPUs is made.
2. Fewer overall weight updates are made, which saves compute time.

Here is the pipeline outline extended to show the progress of 16 mini-batches 
followed by a weight update. Notice that the best utilization of the IPUs is 
during the main phase and that this is sustained until the last mini-batch enters 
the pipeline, following which the ramp down begins. Also notice that weight 
updates are only applied once, following the ramp down (after the pipeline has 
been drained of all mini-batches.)

![Execution Phases](images/execution_phases.png)
"""
"""
## Tutorial Walkthrough

This tutorial starts with a simple multi-stage model that trains on the MNIST dataset.  

Note:  

* Your 'real-world' models are probably more complicated, but the techniques 
learned in this tutorial can still be applied.  
* The results you see locally will depend on which IPU hardware you run on and 
which SDK you are using.  
* Similarly, when looking at profiling reports you have generated yourself, 
they may not look exactly the same as the reports presented in this tutorial.  
* The `scripts/profile.sh` helper script is used to capture profiling reports; 
this will  
  a) override the application parameters to run just a single step without 
  repeat and with constrained batch accumulation,  
  b) enable autoReport to capture all reports,  
  c) enable synthetic data to remove host IO from the execution trace.  
"""
"""
### Tutorial Step 1: The Existing Single IPU Application

The code we will start with can also be found [`step1_single_ipu.py`](step1_single_ipu.py).

Here is a slightly simplified version of it. Take a look at the code and 
familiarise yourself with it. 

We start with importing all necessary libraries
"""

import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.python import ipu

tf.disable_eager_execution()
tf.disable_v2_behavior()

"""
If you see output something like this below then it means that you forgot to 
install the Graphcore TensorFlow 1 wheel. See [the Requirements section](#requirements).

```
Traceback (most recent call last):
  File "step1_single_ipu.py", line 6, in <module>
    import tensorflow.compat.v1 as tf
ModuleNotFoundError: No module named 'tensorflow'
```
"""
"""
Set hyperparameters, if you want to run the script with different ones remember
to rerun all cells below
"""
BATCH_SIZE = 32
REPEAT_COUNT = 160  # The number of times the pipeline will be executed for each step.
EPOCHS = 50
LEARNING_RATE = 0.01
BATCHES_TO_ACCUMULATE = 16  # How many batches to process before processing gradients and updating weights.
"""
Create a function responsible for generating a dataset
"""


def create_dataset(batch_size):
    train_data, _ = mnist.load_data()

    def normalise(x, y):
        return x.astype("float32") / 255.0, y.astype("int32")

    x_train, y_train = normalise(*train_data)

    def generator():
        return zip(x_train, y_train)

    types = (x_train.dtype, y_train.dtype)
    shapes = (x_train.shape[1:], y_train.shape[1:])

    num_examples = len(x_train)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    # Use 'drop_remainder=True' because XLA (and the compiled static IPU graph)
    # expect a complete, fixed sized, set of data as input.
    # Caching and prefetching are important to prevent the host data
    # feed from being the bottleneck for throughput.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(num_examples)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return num_examples, dataset


num_examples, dataset = create_dataset(batch_size=BATCH_SIZE)
num_train_examples = int(EPOCHS * num_examples)

"""
Create the data queues from/to IPU
"""
infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
"""
With batch size BS and repeat count RPT, at every step n = (BS * RPT) examples 
are used. Ensure we process a whole multiple of the batch accumulation count.
"""
remainder = REPEAT_COUNT % BATCHES_TO_ACCUMULATE
if remainder > 0:
    REPEAT_COUNT += BATCHES_TO_ACCUMULATE - remainder
    print(f'Rounding up repeat count to whole multiple of '
          f'batches-to-accumulate (== {REPEAT_COUNT})')
examples_per_step = BATCH_SIZE * REPEAT_COUNT
"""
In order to evaluate at least N total examples, do ceil(N / n) steps
"""
steps = (num_train_examples + examples_per_step - 1) // examples_per_step
training_samples = steps * examples_per_step
print(f'Steps {steps} x examples per step {examples_per_step} '
      f'(== {training_samples} training examples, {training_samples / num_examples} '
      f'epochs of {num_examples} examples)')
"""
Now we will compile the learning rate and create the model 
"""
with tf.device('cpu'):
    learning_rate = tf.placeholder(np.float32, [])
"""
"""


def model(learning_rate, images, labels):
    # Receiving images,labels (x args.batch_size) via infeed.
    # The scoping here helps clarify the execution trace when using --profile.
    with tf.variable_scope("flatten"):
        activations = layers.Flatten()(images)
    with tf.variable_scope("dense256"):
        activations = layers.Dense(256, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense128"):
        activations = layers.Dense(128, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense64"):
        activations = layers.Dense(64, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense32"):
        activations = layers.Dense(32, activation=tf.nn.relu)(activations)
    with tf.variable_scope("logits"):
        logits = layers.Dense(10)(activations)
    with tf.variable_scope("softmax_ce"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
    with tf.variable_scope("mean"):
        loss = tf.reduce_mean(cross_entropy)
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        if BATCHES_TO_ACCUMULATE > 1:
            optimizer = ipu.optimizers.GradientAccumulationOptimizerV2(
                optimizer, num_mini_batches=BATCHES_TO_ACCUMULATE)
        train_op = optimizer.minimize(loss=loss)
    return learning_rate, train_op, outfeed_queue.enqueue(loss)


"""
Create functions that runs the training step `REPEAT_COUNT` times by iterating 
the model in an IPU repeat loop, then compile the model.
"""


def loop_repeat_model(learning_rate):
    r = ipu.loops.repeat(REPEAT_COUNT, model, [learning_rate], infeed_queue)
    return r


with ipu.scopes.ipu_scope("/device:IPU:0"):
    compiled_model = ipu.ipu_compiler.compile(loop_repeat_model,
                                              inputs=[learning_rate])
outfeed_op = outfeed_queue.dequeue()
""""
Configure the IPU.
"""

ipu.utils.move_variable_initialization_to_cpu()
init_op = tf.global_variables_initializer()

ipu_configuration = ipu.config.IPUConfig()
ipu_configuration.auto_select_ipus = 1
ipu_configuration.configure_ipu_system()

"""
We are ready to start the training process!
"""


def train():
    with tf.Session() as sess:
        # Initialize
        sess.run(init_op)
        sess.run(infeed_queue.initializer)
        # Run
        begin = time.time()
        for step in range(steps):
            sess.run(compiled_model, {learning_rate: LEARNING_RATE})
            # Read the outfeed for the training losses
            losses = sess.run(outfeed_op)
            if losses is not None and len(losses):
                epoch = float(examples_per_step * step / num_examples)
                if step == (steps - 1) or (step % 10) == 0:
                    print("Step {}, Epoch {:.1f}, Mean loss: {:.3f}".format(
                        step, epoch, np.mean(losses)))
        end = time.time()
        elapsed = end - begin
        samples_per_second = training_samples / elapsed
        print("Elapsed {:.2f}, {:.2f} samples/sec".format(elapsed,
                                                          samples_per_second))


train()
print("Stage 1 ran successfully")
# sst_hide_output
"""
You can also run it with `$ python3 step1_single_ipu.py`  

The model is running on a single IPU without pipelining. You should see it train 
with output similar to this below:

```
$ python3 step1_single_ipu.py
<CUT>
Steps 586 x examples per step 5120 (== 3000320 training examples, 50.00 epochs of 60000 examples)
<CUT>
Step 0, Epoch 0.0, Mean loss: 2.157
Step 10, Epoch 0.9, Mean loss: 0.351
<CUT>
Step 585, Epoch 49.9, Mean loss: 0.001
Elapsed <CUT>
```
"""
"""
This is the model outline:

![Model Schematic](images/model_schematic.png)

The loss is optimized using `GradientDescentOptimizer` and `GradientAccumulationOptimizerV2`:

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
if args.batches_to_accumulate > 1:
    optimizer = ipu.optimizers.GradientAccumulationOptimizerV2(
            optimizer, num_mini_batches=args.batches_to_accumulate)
train_op = optimizer.minimize(loss=loss)
```

The `GradientAccumulationOptimizerV2` is a wrapper for an optimizer where 
instead of performing the weight update for every batch, gradients across 
multiple batches are accumulated. After multiple batches have been processed, 
their accumulated gradients are used to compute the weight update. The 
effective batch size is the product of the model batch size and the gradient 
accumulation count. The `GradientAccumulationOptimizerV2` optimizer can be 
used to wrap any other TensorFlow optimizer. In this case it is wrapping 
a `GradientDescentOptimizer`. 

See the TensorFlow 1 API documentation for details: [GradientAccumulationOptimizerV2](<https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2>)
"""
"""
Generate a profile report into directory `./profile_step1_single_ipu` with:

`$ scripts/profile.sh step1_single_ipu.py`

Use PopVision Graph Analyser to view the execution trace.
Remember that the `profile.sh` script limits the execution to a single batch.

If this is your first time using the PopVision Graph Analyser, then see the user 
guide: [PopVision User Guide](<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/index.html#popvision-user-guide>)

You should see something like this (zoomed in to show a single mini-batch):

![Execution trace](images/step1_single_ipu_execution_trace.png)

The key points to note are:

* IPU0 runs all layers from 'flatten' to 'softmax_ce' and the optimizer.  
* Because `GradientAccumulationOptimizerV2` is being used, the gradient descent 
is deferred until `args.batches_to_accumulate` mini-batches have been processed.
* The application uses `with tf.variable_scope(...)` to declare a context manager
for each layer; variables and ops inherit the scope name, which provides useful 
context in the Graph Analyser.

Scroll to the far right in Graph Analyser to see the gradient descent step:

![Execution trace - Gradient Descent](images/step1_single_ipu_execution_trace_gradient_descent.png)
"""
"""
# Tutorial Step 2: Running The Model On Multiple IPUs Using Sharding

Let's look at how we can shard a model to run it on multiple IPUs **without**
pipelining.

This model outline shows how the operations will be allocated to shards:

![Model Schematic (sharded)](images/model_schematic_sharded.png)
"""
"""
### Tutorial Step 2: Code Changes
Now we will modify the model code so it will be divided between two shards.
In the beginning we add two sharding scopes:
- `ipu.scopes.ipu_shard(0) - this will be the context for the set of layers 
that will end up running on IPU0
- `ipu.scopes.ipu_shard(1) - this will be the context for the set of layers 
that will end up running on IPU1

Then, we split the model across the two scopes, so that IPU0 runs layers 
'flatten' to 'dense64' and IPU1 runs layers ' dense32' to 'softmax_ce' plus 
the optimizer.

After this change is applied, the model definition function should look like
this:
"""


def model(learning_rate, images, labels):
    # Receiving images,labels (x args.batch_size) via infeed.
    # The scoping here helps clarify the execution trace when using --profile.

    with ipu.scopes.ipu_shard(0):
        with tf.variable_scope("flatten"):
            activations = layers.Flatten()(images)
        with tf.variable_scope("dense256"):
            activations = layers.Dense(256, activation=tf.nn.relu)(activations)
        with tf.variable_scope("dense128"):
            activations = layers.Dense(128, activation=tf.nn.relu)(activations)
        with tf.variable_scope("dense64"):
            activations = layers.Dense(64, activation=tf.nn.relu)(activations)
    with ipu.scopes.ipu_shard(1):
        with tf.variable_scope("dense32"):
            activations = layers.Dense(32, activation=tf.nn.relu)(activations)
        with tf.variable_scope("logits"):
            logits = layers.Dense(10)(activations)
        with tf.variable_scope("softmax_ce"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        with tf.variable_scope("mean"):
            loss = tf.reduce_mean(cross_entropy)
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
            if BATCHES_TO_ACCUMULATE > 1:
                optimizer = ipu.optimizers.GradientAccumulationOptimizerV2(
                    optimizer,
                    num_mini_batches=BATCHES_TO_ACCUMULATE)
            train_op = optimizer.minimize(loss=loss)
        # A control dependency is used here to ensure that
        # the train_op is not removed.
        with tf.control_dependencies([train_op]):
            return learning_rate, outfeed_queue.enqueue(loss)


"""
We also have to increase the IPU count from 1 to 2.
"""
ipu_configuration = ipu.config.IPUConfig()
ipu_configuration.auto_select_ipus = 2
ipu_configuration.configure_ipu_system()
"""
We are ready to compile model again and start the training process:
"""

infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

with ipu.scopes.ipu_scope("/device:IPU:0"):
    compiled_model = ipu.ipu_compiler.compile(loop_repeat_model,
                                              inputs=[learning_rate])

outfeed_op = outfeed_queue.dequeue()

ipu.utils.move_variable_initialization_to_cpu()
init_op = tf.global_variables_initializer()

train()
print("Stage 2 ran successfully")
# sst_hide_output
"""
You should see it train with output something like this below.

```
$ python3 step2_sharding.py
<CUT>
Steps 586 x examples per step 5120 (== 3000320 training examples, 50.00 epochs of 60000 examples)
<CUT>
Step 0, Epoch 0.0, Mean loss: 2.184
Step 10, Epoch 0.9, Mean loss: 0.362
<CUT>
Step 585, Epoch 49.9, Mean loss: 0.001
Elapsed <CUT>
```

Complete working example also can be found in `answers/step2_sharding.py`,
you can run the application with the following shell command:

`$ python3 step2_sharding.py`
"""
"""
Generate a profile report into directory `./profile_step2_sharding` with:

`$ scripts/profile.sh step2_sharding.py`

Use PopVision Graph Analyser to view the execution trace.

You should see something like this (zoomed in to show a single mini-batch):

![Execution trace](images/step2_sharding_execution_trace.png)

The key points to note are:

* IPU0 runs layers 'flatten' to 'dense64'.
* IPU1 runs layers 'dense32' to 'softmax_ce' and the optimizer.
* It is not efficient because execution is serialised (there is poor
  utilisation).
* Because `GradientAccumulationOptimizerV2` is being used, the gradient descent
  is deferred until `args.batches_to_accumulate` mini-batches have been
  processed.
* In this specific captured example, the gradients are calculated entirely on
  IPU1.

Also, note that for a small model such as this one, that fits on a single IPU,
sharding does not bring any performance advantages. In fact, because data
exchange between IPUs is slower than data exchange within an IPU, the
performance will be worse than that without sharding.

The completed code for this step can be found
here: [`answers/step2_sharding.py`](answers/step2_sharding.py)
"""