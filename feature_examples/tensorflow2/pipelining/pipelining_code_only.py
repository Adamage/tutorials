# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Number of samples per batch.
BATCH_SIZE = 32

# Number of steps to run per execution. The number of batches to run for
# each TensorFlow function call. At most it would execute a full epoch.
STEPS_PER_EXECUTION = 500

# Number of steps per epoch. The total number of steps (batches of samples)
# for one epoch to finish and starting the next one. The default `None` is
# equal to the number of samples divided by the batch size.
STEPS_PER_EPOCH = STEPS_PER_EXECUTION

# Number of epochs
EPOCHS = 4

# Optimizer parameters.
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Number of devices that will be attached to this model for training and
# inference.
NUM_IPUS = 2

# Number of steps for which the gradients should be accumulated, for each
# configured replica.
STEPS_PER_REPLICA = 4

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu
from tensorflow.keras import Model
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input

def create_dataset(batch_size: int, repeat=True):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(
        lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32))
    )
    if repeat:
        return train_ds.repeat()
    else:
        return train_ds


train_ds = create_dataset(batch_size=BATCH_SIZE)

def make_ipu_config(
        num_ipus: int,
        selection_order: Optional[ipu.utils.SelectionOrder] = None
) -> ipu.config.IPUConfig:

    ipu_configuration = ipu.config.IPUConfig()
    ipu_configuration.auto_select_ipus = num_ipus

    if selection_order:
        ipu_configuration.selection_order = selection_order

    ipu_configuration.configure_ipu_system()
    return ipu_configuration

def train(strategy,
          model_factory,
          train_ds,
          steps_per_replica: int = STEPS_PER_REPLICA,
          steps_per_execution: int = STEPS_PER_EXECUTION,
          steps_per_epoch: int = STEPS_PER_EPOCH,
          epochs: int = 4):

    with strategy.scope():
        model = model_factory()

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM
            ),
            steps_per_execution=steps_per_execution
        )

        if steps_per_replica:
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=steps_per_replica
            )

        model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=epochs)

def create_functional_model(batch_size=BATCH_SIZE):
    input_layer = Input(
        shape=(28, 28, 1),
        dtype=tf.float32,
        batch_size=batch_size
    )
    x = Flatten(name='flatten')(input_layer)
    x = Dense(256, activation='relu', name="dense256")(x)
    x = Dense(128, activation='relu', name="dense128")(x)
    x = Dense(64, activation='relu', name="dense64")(x)
    x = Dense(32, activation='relu', name="dense32")(x)
    x = Dense(10, name="logits")(x)

    model = Model(
        inputs=input_layer,
        outputs=x,
        name="singleIPU"
    )
    return model

ipu_configuration = make_ipu_config(num_ipus=1)

train(
    strategy=ipu.ipu_strategy.IPUStrategy(),
    model_factory=create_functional_model,
    train_ds=train_ds
)

def create_sequential_model():
    seq_model = Sequential(
        layers=[
            Flatten(name='flatten'),
            Dense(256, activation='relu', name="dense256"),
            Dense(128, activation='relu', name="dense128"),
            Dense(64, activation='relu', name="dense64"),
            Dense(32, activation='relu', name="dense32"),
            Dense(10, activation='softmax', name="logits")
        ],
        name="singleIPU"
    )
    return seq_model

ipu_configuration = make_ipu_config(num_ipus=1)

train(
    strategy=ipu.ipu_strategy.IPUStrategy(),
    model_factory=create_sequential_model,
    train_ds=train_ds
)

def create_functional_model_with_stages():
    input_layer = Input(shape=(28, 28, 1),
                                     dtype=tf.float32,
                                     batch_size=BATCH_SIZE)
    with ipu.keras.PipelineStage(0):
        x = Flatten(name='flatten')(input_layer)
        x = Dense(256, activation='relu', name="dense256")(x)
        x = Dense(128, activation='relu', name="dense128")(x)
        x = Dense(64, activation='relu', name="dense64")(x)

    with ipu.keras.PipelineStage(1):
        x = Dense(32, activation='relu', name="dense32")(x)
        x = Dense(10, name="logits")(x)

    model = Model(inputs=input_layer,
                  outputs=x,
                  name="multipleIPUfunctional")
    return model

ipu_configuration = make_ipu_config(num_ipus=2)

train(
    strategy=ipu.ipu_strategy.IPUStrategy(),
    model_factory=create_functional_model_with_stages,
    train_ds=train_ds
)

def create_pipeline_sequential_model():
    seq_model = Sequential(
        layers=[
            Flatten(name='flatten'),
            Dense(256, activation='relu', name="dense256"),
            Dense(128, activation='relu', name="dense128"),
            Dense(64, activation='relu', name="dense64"),
            Dense(32, activation='relu', name="dense32"),
            Dense(10, activation='softmax', name="logits")
        ],
        name="multipleIPUsequential"
    )
    seq_model.set_pipeline_stage_assignment([0, 0, 1, 1, 1, 1])

    return seq_model

ipu_configuration = make_ipu_config(num_ipus=2)

train(
    strategy=ipu.ipu_strategy.IPUStrategy(),
    model_factory=create_pipeline_sequential_model,
    train_ds=train_ds
)

def create_pipeline_sequential_model_interleaved():
    seq_model = Sequential(
        layers=[
            Flatten(name='flatten'),
            Dense(256, activation='relu', name="dense256"),
            Dense(128, activation='relu', name="dense128"),
            Dense(64, activation='relu', name="dense64"),
            Dense(32, activation='relu', name="dense32"),
            Dense(10, activation='softmax', name="logits")
        ],
        name="multipleIPUsequential"
    )
    seq_model.set_pipeline_stage_assignment([0, 0, 1, 1, 1, 1])

    seq_model.set_pipelining_options(
        schedule=ipu.ops.pipelining_ops.PipelineSchedule.Interleaved
    )
    return seq_model

ipu_configuration = make_ipu_config(num_ipus=2)

train(
    strategy=ipu.ipu_strategy.IPUStrategy(),
    model_factory=create_pipeline_sequential_model_interleaved,
    train_ds=train_ds
)
