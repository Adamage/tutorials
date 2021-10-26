# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from outfeed_callback import OutfeedCallback
from outfeed_optimizer import OutfeedOptimizer, OutfeedOptimizerMode
import outfeed_layers
from outfeed_wrapper import MaybeOutfeedQueue

def create_dataset():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    train_ds = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)) \
        .batch(32, drop_remainder=True)

    train_ds = train_ds.map(
        lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32))
    )
    return train_ds

def create_pipeline_sequential_model(multi_activations_outfeed_queue):
    seq_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', name="Dense_256"),
        keras.layers.Dense(128, activation='relu', name="Dense_128"),
        outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                    final_outfeed=False,
                                    name="Dense_128_acts"),
        keras.layers.Dense(10, activation='softmax', name="Dense_10"),
        outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                    final_outfeed=True,
                                    name="Dense_10_acts")
    ])
    seq_model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
    seq_model.set_pipeline_stage_assignment([0, 0, 1, 1, 1, 1])
    return seq_model

# [boolean] Should the code outfeed the pre-accumulated gradients, rather than
# accumulated gradients? Only makes a difference when using gradient
# accumulation, which is always the case when pipelining is enabled.
outfeed_pre_accumulated_gradients = False

# Number of steps to run per execution. The number of batches to run for
# each TensorFlow function call. At most it would execute a full epoch.
steps_per_execution = 500

# Number of steps per epoch. The total number of steps (batches of samples)
# for one epoch to finish and starting the next one. The default `None` is
# equal to the number of samples divided by the batch size.
steps_per_epoch = steps_per_execution

# Number of epochs
epochs = 3

# [List] String values representing which gradients to add to the dictionary
# that is enqueued on the outfeed queue. Pass `[none]` to disable filtering.
gradients_filters = ['Dense_128']

# [List] Activation filters - strings representing which activations in the
# second `PipelineStage` to add to the dictionary that is enqueued on the
# outfeed queue. Pass `[none]` to disable filtering. Applicable only for
# pipelined models.
activations_filters = ['none']

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--outfeed-pre-accumulated-gradients',
    action='store_true',
    help="Outfeed the pre-accumulated rather than accumulated gradients."
         " Only makes a difference when using gradient accumulation"
         " (which is the case when pipelining).")
parser.add_argument(
    '--steps-per-epoch', type=int, default=500,
    help="Number of steps to run per epoch.")
parser.add_argument(
    '--epochs', type=int, default=3, help="Number of epochs.")
parser.add_argument(
    '--gradients-filters', nargs='+', type=str, default=['Dense_128'],
    help="Space separated strings for determining which gradients"
         " to add to the dict that is enqueued on the outfeed queue."
         " Pass '--gradients-filters none' to disable filtering.")
parser.add_argument(
    '--activations-filters', nargs='+', type=str, default=['none'],
    help="Space separated strings for determining which activations"
         " in the second PipelineStage"
         " to add to the dict that is enqueued on the outfeed queue."
         " Pass '--activations-filters none' to disable filtering."
         " (Only applicable for the Pipeline models.)")

args = parser.parse_args()

print(args)

if args.outfeed_pre_accumulated_gradients:
    outfeed_pre_accumulated_gradients = \
        args.outfeed_pre_accumulated_gradients

steps_per_epoch = args.steps_per_epoch
steps_per_execution - steps_per_epoch
epochs = args.epochs

activations_filters = args.activations_filters
gradients_filters = args.gradients_filters

if outfeed_pre_accumulated_gradients:
    outfeed_optimizer_mode = OutfeedOptimizerMode.AFTER_COMPUTE
else:
    outfeed_optimizer_mode = OutfeedOptimizerMode.BEFORE_APPLY

def process_filters(filters_input):
    if len(filters_input) == 1 and filters_input[0].lower() == "none":
        return None
    return filters_input

def model_with_callbacks(gradients_filters, activations_filters):
    optimizer_q = MaybeOutfeedQueue(filters=process_filters(gradients_filters))
    act_q = MaybeOutfeedQueue(filters=process_filters(activations_filters))

    gradients_cb = OutfeedCallback(outfeed_queue=optimizer_q,
                                   name="Gradients callback")
    multi_layer_cb = OutfeedCallback(outfeed_queue=act_q,
                                     name="Multi-layer activations callback")

    callbacks = [gradients_cb, multi_layer_cb]
    seq_model = create_pipeline_sequential_model(act_q)
    return seq_model, callbacks, optimizer_q

cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.configure_ipu_system()

strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
    seq_model, callbacks, optimizer_outfeed_queue = \
        model_with_callbacks(gradients_filters, activations_filters)

    # Build the graph passing an OutfeedOptimizer to enqueue selected gradients
    seq_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=OutfeedOptimizer(
            wrapped_optimizer=keras.optimizers.SGD(),
            outfeed_queue=optimizer_outfeed_queue,
            outfeed_optimizer_mode=outfeed_optimizer_mode,
            model=seq_model
        ),
        steps_per_execution=steps_per_execution
    )

    # Train the model passing the callbacks to see the gradients
    # and activations stats
    seq_model.fit(
        create_dataset(),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs
    )
