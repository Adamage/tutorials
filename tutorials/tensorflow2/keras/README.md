# Keras tutorial: How to run on IPU

This tutorial provides an introduction on how to run Keras models on IPUs, and 
features that allow you to fully utilise the capability of the IPU. Please 
refer to the [TensorFlow 2 Keras API reference](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#module-tensorflow.python.ipu.keras) 
for full details of all available features.

Requirements:
* Installed and enabled Poplar
* Installed the Graphcore port of TensorFlow 2

Refer to the Getting Started guide for your IPU System for instructions.

#### Directory Structure

* `completed_demos`: Completed versions of the scripts described in this tutorial
* `completed_example`: A completed example of running Keras models on the IPU
* `test`: A directory that contains test scripts

#### Keras MNIST example

The script below (which also can be found in `demo.py`) illustrates a simple 
example using the MNIST numeral dataset, which consists of 60,000 images for 
training and 10,000 images for testing. The images are of handwritten 
digits 0-9, and they must be classified according to which digit they represent. 
MNIST classification is a toy example problem, but is sufficient to outline 
the concepts introduced in this tutorial.

Without changes, the script will run the Keras model on the CPU. It is based 
on the original Keras tutorial and as such is vanilla Keras code. You can run 
this now to see its output. In the following sections, we will go through the 
changes needed to make this run on the IPU.

Running cell below or `python3 demo.py` gives the following throughput values 
for training:

```
Epoch 1/3
938/938 [==============================] - 10s 10ms/step - loss: 1.6732 - accuracy: 0.4536
Epoch 2/3
938/938 [==============================] - 9s 10ms/step - loss: 0.3618 - accuracy: 0.8890
Epoch 3/3
938/938 [==============================] - 9s 10ms/step - loss: 0.2376 - accuracy: 0.9289
```


```python
import tensorflow.keras as keras
import numpy as np

# Store class and shape information.
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64


def prepare_data():
    # Load the MNIST dataset from keras.datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the images.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # When dealing with images, we usually want an explicit channel dimension,
    # even when it is 1.
    # Each sample thus has a shape of (28, 28, 1).
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Finally, convert class assignments to a binary class matrix.
    # Each row can be seen as a rank-1 "one-hot" tensor.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def model_fn():
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add layers to the graph.
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x


print('Keras MNIST example, running on CPU')
(x_train, y_train), (x_test, y_test) = prepare_data()

# Model.__init__ takes two required arguments, inputs and outputs.
model = keras.Model(*model_fn())

# Compile our model with Stochastic Gradient Descent as an optimizer
# and Categorical Cross Entropy as a loss.
model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])

print(model.summary())

print('\nTraining')
model.fit(x_train, y_train, epochs=3, batch_size=batch_size)

print('\nEvaluation')
model.evaluate(x_test, y_test, batch_size=batch_size)
```


#### Running the example on the IPU

In this section, we will make a series of edits to `demo.py` in order to train 
the model using the IPU. Make a copy of `demo.py` to follow along.

##### 1. Import the TensorFlow IPU module

First, we import the TensorFlow IPU module.

Add the following import statement to the beginning of your script:



```python
from tensorflow.python import ipu
```


For the `ipu` module to function properly, we must import it directly rather 
than accessing it through the top-level TensorFlow module.

##### 2. Preparing the dataset

Some extra care must be taken when preparing a dataset for training a Keras 
model on the IPU. The Poplar software stack does not support using tensors 
with shapes which are not known when the model is compiled, so we must make 
sure the sizes of our datasets are divisible by the batch size. We introduce 
a utility function, `make_divisible`, which computes the largest number, no 
larger than a given number, which is divisible by a given divisor. This will be 
of further use later.


```python
def make_divisible(number):
    return number - number % batch_size
```

Adjust dataset lengths to be divisible by the batch size:


```python
def prepare_data_trim_to_size():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    train_data_len = x_train.shape[0]
    train_data_len = make_divisible(train_data_len)
    x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

    test_data_len = x_test.shape[0]
    test_data_len = make_divisible(test_data_len)
    x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)
```

With a batch size of 64, we lose 32 training examples and 48 evaluation 
examples, which is less than 0.2% of each dataset.

There are other ways to prepare a dataset for training on the IPU. You can 
create a `tf.data.Dataset` object using your data, then use its `.repeat()` 
method to create a looped version of the dataset. If you do not want to lose 
any data, you can pad the datasets with tensors of zeros, then set 
`sample_weight` to be a vector of 1’s and 0’s according to which values are 
real so the extra values don’t affect the training process (though this may be 
slower than using the other methods).

##### 3. Add IPU configuration

To use the IPU, you must create an IPU session configuration:


```python
ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = 1
ipu_config.configure_ipu_system()
```

This is all we need to get a small model up and running, though a full list of 
configuration options is available in the [API documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.config.IPUConfig).

##### 4. Specify IPU strategy

Next, add the following code:


```python
# Create an execution strategy.
strategy = ipu.ipu_strategy.IPUStrategy()
```

The `tf.distribute.Strategy` is an API to distribute training across multiple 
devices. `IPUStrategy` is a subclass which targets a system with one or more 
IPUs attached. Another subclass, [IPUMultiWorkerStrategy](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy), 
targets a multi-system configuration.

##### 5. Wrap the model within the IPU strategy scope

Creating variables and Keras models within the scope of the `IPUStrategy` 
object will ensure that they are placed on the IPU. To do this, we create a 
`strategy.scope()` context manager and move all the model code inside it:


```python
print('Keras MNIST example, running on IPU')
(x_train, y_train), (x_test, y_test) = prepare_data_trim_to_size()

with strategy.scope():
    model = keras.Model(*model_fn())

    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])

    model.summary()
    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=64)

    print('\nEvaluation')
    model.evaluate(x_test, y_test)
```

Note that the function `model_fn()` can be readily reused, and all we really 
need to do is move the code inside the context of `strategy.scope()`. Prior to 
the release of version 2.2.0 of the Poplar SDK, it would have been necessary to 
make the model an instance of the `ipu.keras.Model` class, which has been 
removed as of version 2.2.0.

While all computation will now be performed on the IPU, the initialisation of 
variables will still be performed on the host.


##### 6. Results

Running the code gives the following throughput values for training:

```
937/937 [==============================] - 45s 3ms/step - loss: 1.5260 - accuracy: 0.4949
Epoch 2/3
937/937 [==============================] - 2s 3ms/step - loss: 0.3412 - accuracy: 0.8968
Epoch 3/3
937/937 [==============================] - 3s 3ms/step - loss: 0.2358 - accuracy: 0.9294
```

The training time has been significantly reduced by use of the IPU. We ignore 
the reported total for the first epoch because this time includes the model's 
compilation time.

The file `completed_demos/completed_demo_ipu.py` shows what the code looks like 
after the above changes are made. 

#### Going faster by setting `steps_per_execution`

The IPU implementation above is fast, but not as fast as it could be. This is 
because, unless we specify otherwise, the program that runs on the IPU will 
only process a single batch, so we cannot get a speedup from loading the data 
asynchronously and using a looped version of this program.

To change this, we must set the `steps_per_execution` argument in 
`model.compile()`. This sets the number of batches processed in each execution 
of the underlying IPU program.

Now not only the number of data must divide equally into all batches, but also 
the number of batches must divide into the number of steps, for this purpose 
we will overload the make_divisible function:


```python
def make_divisible(number):
    steps_per_execution = number // batch_size
    return number - number % (batch_size * steps_per_execution)
```

The number of examples in the dataset must be divisible by the number of 
examples processed per execution (that is, `steps_per_execution * batch_size`). 
Here, we set `steps_per_execution` to be `(length of dataset) // batch_size` 
for maximum throughput and so that we do not lose any more data than we have to, 
though this code should work just as well with a different, smaller value.

Now we update the code from `with strategy.scope():` onwards by passing 
`steps_per_execution` as an argument to `model.compile()`, and providing our 
`batch_size` value to `model.fit()` and `model.evaluate()`. We can re-compile 
the model with a different value of `steps_per_execution` between running 
`model.fit()` and `model.evaluate()`, so we do so here, although it isn't 
compulsory.


```python
print('Keras MNIST example, running on IPU with steps_per_execution')
(x_train, y_train), (x_test, y_test) = prepare_data_trim_to_size()

with strategy.scope():
    model = keras.Model(*model_fn())

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=len(x_train) // batch_size)

    model.summary()
    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=64)

    print('\nEvaluation')
    model.evaluate(x_test, y_test)

    model.summary()
    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=batch_size)
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=len(x_test) // batch_size)

    print('\nEvaluation')
    model.evaluate(x_test, y_test, batch_size=batch_size)
```


Running this code, the model trains much faster:

```
937/937 [==============================] - 43s 46ms/step - loss: 1.0042 - accuracy: 0.6783
Epoch 2/3
937/937 [==============================] - 0s 224us/step - loss: 0.3021 - accuracy: 0.9079
Epoch 3/3
937/937 [==============================] - 0s 222us/step - loss: 0.2240 - accuracy: 0.9326
```

The file `completed_demos/completed_demo_faster.py` shows what the code looks 
like after the above changes are made.


#### Replication

Another way to speed up the training of a model is to make a copy of the model 
on each of multiple IPUs, updating the parameters of the model on all IPUs after 
each forward and backward pass. This is called _replication_, and can be 
done in Keras with very few code changes. 

First, we'll add variables for the number of IPUs and the number of replicas:


```python
num_ipus = num_replicas = 2
```

Because our model is written for one IPU, the number of replicas will be equal 
to the number of IPUs.

We will need to adjust for the fact that with replication, a batch is processed 
on each replica for each step, so `steps_per_execution` needs to be divisible 
by the number of replicas. Also, the maximum value of `steps_per_execution` is 
now `train_data_len // (batch_size * num_replicas)`, since the number of 
examples processed in each step is now `(batch_size * num_replicas)`. 
We therefore add two lines to the dataset-adjustment code:


```python
def make_divisible(number):
    return number - number % (batch_size * num_replicas)
```


We'll need to acquire multiple IPUs, so we update the configuration step:



```python
ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = num_ipus
ipu_config.configure_ipu_system()
```

    WARNING:tensorflow:Resetting existing IPU configuration before applying new configuration


These are all the changes we need to make to replicate the model and train on 
multiple IPUs. There is no need to explicitly copy the model or organise the 
exchange of weight updates between the IPUs because all of these details are 
handled automatically, as long as we select multiple IPUs and create and use 
our model within the scope of an `IPUStrategy` object.


```python
print('Keras MNIST example, running on IPU with replication')
(x_train, y_train), (x_test, y_test) = prepare_data_trim_to_size()

with strategy.scope():
    model = keras.Model(*model_fn())

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    train_steps = make_divisible(len(x_train))
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=train_steps // (batch_size * num_replicas))

    model.summary()
    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=64)

    print('\nEvaluation')
    model.evaluate(x_test, y_test)

    model.summary()
    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=batch_size)

    test_steps = make_divisible(len(x_test))
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=test_steps // (batch_size * num_replicas))

    print('\nEvaluation')
    model.evaluate(x_test, y_test, batch_size=batch_size)
```

With replication, the model trains even faster:

```
936/936 [==============================] - 44s 47ms/step - loss: 1.1886 - accuracy: 0.6213
Epoch 2/3
936/936 [==============================] - 0s 135us/step - loss: 0.3155 - accuracy: 0.9054
Epoch 3/3
936/936 [==============================] - 0s 134us/step - loss: 0.2277 - accuracy: 0.9304
```
However, we do not get a perfect 2x speedup because the gradients must be 
exchanged between the IPUs before each weight update.

The file `completed_demos/completed_demo_replicated.py` shows what the code 
looks like after the above changes are made. 


#### Pipelining

Pipelining can also be enabled to split a Keras model across multiple IPUs. A 
pipelined model will execute multiple sections (called _stages_) of a model on 
individual IPUs concurrently by pipelining mini-batches of data through the stages.

One of the key features of pipelining on the IPU is _gradient accumulation_. 
Forward and backward passes will be performed on several batches without 
performing a weight update. Instead, a cumulative sum of the gradients is 
updated after each forward and backward pass, and the weight update is applied 
only after a certain number of batches have been processed. This helps ensure 
consistency between the weights used in the forward and backward passes, and 
can be used to train with a batch size that wouldn't fit otherwise. To learn 
more about the specifics of pipelining, you can read [the relevant section of 
the Technical Note on Model Parallelism in TensorFlow](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html).

In this final part of the tutorial, we will pipeline our model over two stages. 
We will need to change the value of `num_replicas`, and create a variable for 
the number of gradient accumulation steps per replica:


```python
num_ipus = 2
num_replicas = num_ipus // 2
gradient_accumulation_steps_per_replica = 8
```

The number of gradient accumulation steps is the number of batches for which we 
perform the forward and backward passes before performing a weight update. 

There are multiple ways to execute a pipeline, called _schedules_. The grouped 
and interleaved schedules are the most efficient because they execute stages 
in parallel, while the sequential schedule is mostly used for debugging. 
 In this tutorial, we will use the grouped schedule, which is the default.

When using the grouped schedule, `gradient_accumulation_steps_per_replica` 
must be divisible by `(number of pipeline stages) * 2`. When using the 
interleaved schedule, `gradient_accumulation_steps_per_replica` must be 
divisible by `(number of pipeline stages)`. You can read more about the 
specifics of the different pipeline schedules in [the relevant section of the 
technical note on Model parallelism with TensorFlow](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html#pipeline-scheduling).

If we use more than two IPUs, the model will be automatically replicated to 
fill up the requested number of IPUs. For example, if we select 8 IPUs for our 
2-IPU model, four replicas of the model will be produced.

We also need to adjust `steps_per_execution` to be divisible by the total number 
of gradient accumulation steps across 
all replicas, so we make a slight change to the dataset-adjusting code:


```python
def make_divisible(number):
    return number - number % (batch_size * num_replicas * gradient_accumulation_steps_per_replica)
```

When defining a model using the Keras Functional API, we control what parts of 
the model go into which stages with the `PipelineStage` context manager. 
Replace the model implementation in `demo.py` with:



```python
def model_fn_pipielines():
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add graph nodes for the first pipeline stage.
    with ipu.keras.PipelineStage(0):
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)

    # Add graph nodes for the second pipeline stage.
    with ipu.keras.PipelineStage(1):
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x
```

Any operations created inside a `PipelineStage(x)` context manager will be 
placed in the `x`th pipeline stage (where the stages are numbered starting from 0). 
Here, the model has been divided into two pipeline stages that run concurrently.

If you define your model using the Keras Sequential API, you can use the 
model's `set_pipeline_stage_assignment` method to assign pipeline stages to layers.

Now all we need to do is configure the pipelining-specific aspects of our model. 
Add the following line just before the first call to `model.compile()`:




```python
print('Keras MNIST example, running on IPU with pipelining')
(x_train, y_train), (x_test, y_test) = prepare_data_trim_to_size()

with strategy.scope():
    model = keras.Model(*model_fn_pipielines())

    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica,
        pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped
    )

    train_steps_per_execution = make_divisible(len(x_train)) // (batch_size * num_replicas)
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=train_steps_per_execution)

    print(model.summary())

    print('\nTraining')

    model.fit(x_train, y_train, epochs=3, batch_size=batch_size)

    print('\nEvaluation')
    test_steps_per_execution = make_divisible(len(x_test)) // (batch_size * num_replicas)
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"],
                  steps_per_execution=test_steps_per_execution)

    model.evaluate(x_test, y_test, batch_size=batch_size)
```


Within the scope of an `IPUStrategy`, IPU-specific methods such as 
`set_pipelining_options` are dynamically added to the base `keras.Model` class, 
which allows us to configure IPU-specific aspects of the model. 
We could use the interleaved schedule here by changing `Grouped` to `Interleaved`.

The file `completed_demos/completed_demo_pipelining.py` shows what the code 
looks like after the above changes are made. 

#### Completed example

The folder `completed_example` contains a complete implementation of the 
illustrated Keras model which is more easily 
configured than the scripts in the `completed_demos` directory. This has been 
provided for you to experiment with. Run `python3 completed_example/main.py` 
to run the standard Keras model on a CPU.

The `--use-ipu` and `--pipelining` flags allow you to run the Keras model on the 
IPU and (optionally) adopt the 
pipelining feature respectively. The gradient accumulation count can be adjusted 
with the `--gradient-accumulation-count` flag.

Note that the code in `completed_example` has been refactored into 3 parts:
* `main.py`: Main code to be run.
* `model.py`: Implementation of a standard Keras model and a pipelined Keras model.
* `utils.py`: Contains functions that load the data and argument parser.


#### License
This example is licensed under the Apache License 2.0 - see the LICENSE file in 
this directory.

This directory contains derived work from the following:

Keras simple MNIST convnet example: https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py

Licensed under the Apache License, Version 2.0 (the "License"); you may not use 
this file except in compliance with the License. You may obtain a copy of the 
License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations 
under the License.
