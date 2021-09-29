#!/usr/bin/env python3
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

    # When dealing with images, we usually want an explicit channel dimension, even when it is 1.
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

from tensorflow.python import ipu

def make_divisible(number):
    return number - number % batch_size

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

ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = 1
ipu_config.configure_ipu_system()

# Create an execution strategy.
strategy = ipu.ipu_strategy.IPUStrategy()

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

def make_divisible(number):
    steps_per_execution = number // batch_size
    return number - number % (batch_size * steps_per_execution)

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

num_ipus = num_replicas = 2

def make_divisible(number):
    print(" I AM ALIVE AGIAN")
    return number - number % (batch_size * num_replicas)

ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = num_ipus
ipu_config.configure_ipu_system()

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

num_ipus = 2
num_replicas = num_ipus // 2
gradient_accumulation_steps_per_replica = 8

def make_divisible(number):
    return number - number % (batch_size * num_replicas * gradient_accumulation_steps_per_replica)

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
