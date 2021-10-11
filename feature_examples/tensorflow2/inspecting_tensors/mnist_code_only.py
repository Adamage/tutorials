# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from outfeed_callback import OutfeedCallback
from outfeed_optimizer import OutfeedOptimizer, OutfeedOptimizerMode
import outfeed_layers
from outfeed_wrapper import MaybeOutfeedQueue

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")

def create_dataset():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)) \
        .batch(32, drop_remainder=True)

    train_ds = train_ds.map(
        lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32))
    )
    return train_ds

def create_model(
        activations_outfeed_queue,
        gradient_accumulation_steps_per_replica
):
    input_layer = keras.layers.Input(
        shape=(28, 28, 1),
        dtype=tf.float32,
        batch_size=32
    )
    x = keras.layers.Flatten()(input_layer)
    x = keras.layers.Dense(128, activation='relu', name="Dense_128")(x)

    # Outfeed the activations for a single layer:
    x = outfeed_layers.Outfeed(
        activations_outfeed_queue,
        name="Dense_128_acts")(x)

    x = keras.layers.Dense(10, activation='softmax',  name="Dense_10")(x)

    keras_model = keras.Model(input_layer, x)
    keras_model.set_gradient_accumulation_options(
        gradient_accumulation_steps_per_replica=
        gradient_accumulation_steps_per_replica
    )
    return keras_model

def create_pipeline_model(
        multi_activations_outfeed_queue,
        gradient_accumulation_steps_per_replica
):
    input_layer = keras.layers.Input(shape=(28, 28, 1),
                                     dtype=tf.float32,
                                     batch_size=32)

    with ipu.keras.PipelineStage(0):
        x = keras.layers.Flatten()(input_layer)
        x = keras.layers.Dense(256, activation='relu', name="Dense_256")(x)

    with ipu.keras.PipelineStage(1):
        x = keras.layers.Dense(128, activation='relu', name="Dense_128")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                        final_outfeed=False,
                                        name="Dense_128_acts")(x)
        x = keras.layers.Dense(10, activation='softmax',  name="Dense_10")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                        final_outfeed=True,
                                        name="Dense_10_acts")(x)
    model = keras.Model(input_layer, x)
    model.set_pipelining_options(gradient_accumulation_steps_per_replica=
                                 gradient_accumulation_steps_per_replica)
    return model

def create_sequential_model(
        activations_outfeed_queue,
        gradient_accumulation_steps_per_replica
):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', name="Dense_128"),
        outfeed_layers.Outfeed(activations_outfeed_queue, name="Dense_128_acts"),
        keras.layers.Dense(10, activation='softmax', name="Dense_10")
    ])
    model\
        .set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=
          gradient_accumulation_steps_per_replica
        )
    return model

def create_pipeline_sequential_model(
        multi_activations_outfeed_queue,
        gradient_accumulation_steps_per_replica
):
    model = keras.Sequential([
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
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=
        gradient_accumulation_steps_per_replica
    )
    model.set_pipeline_stage_assignment([0, 0, 1, 1, 1, 1])
    return model

# Model Type can be "Model" or "Sequential"
SEQUENTIAL_MODEL = "Sequential"
REGULAR_MODEL = "Model"
model_type = SEQUENTIAL_MODEL

# [boolean] Should IPU pipelining be disabled?
no_pipelining = False

# [boolean] Should gradient accumulation be enabled? It's always enabled
# for pipelined models.
use_gradient_accumulation = True

# [boolean] Should the code outfeed the pre-accumulated gradients, rather than
# accumulated gradients? Only makes a difference when using gradient
# accumulation, which is always the case when pipelining is enabled.
outfeed_pre_accumulated_gradients = False

# Number of steps to run per epoch.
steps_per_epoch = 500

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

if no_pipelining:
    num_ipus = 1
else:
    num_ipus = 2
    use_gradient_accumulation = True

gradient_accumulation_steps_per_replica = 4

if outfeed_pre_accumulated_gradients:
    outfeed_optimizer_mode = OutfeedOptimizerMode.AFTER_COMPUTE
else:
    outfeed_optimizer_mode = OutfeedOptimizerMode.BEFORE_APPLY

def process_filters(filters_input):
    if len(filters_input) == 1 and filters_input[0].lower() == "none":
        return None
    return filters_input

def instantiate_selected_model_type(
        gradient_accumulation_steps_per_replica,
        no_pipelining,
        use_gradient_accumulation,
        model_type,
        gradients_filters,
        activations_filters
):
    # Create the outfeed queue for selected gradients.
    # Remove the filters to get the gradients for all layers or pass different
    # strings to the argument to select other layer(s)
    optimizer_q = MaybeOutfeedQueue(filters=process_filters(gradients_filters))

    # Create a callback for the gradients.
    gradients_cb = OutfeedCallback(optimizer_q, name="Gradients callback")

    # Create callbacks for the activations in the custom layers.
    activations_q = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    layer_cb = OutfeedCallback(activations_q,
                               name="Single layer activations callback")

    multi_activations_q = MaybeOutfeedQueue(filters=process_filters(
                                                      activations_filters))
    multi_layer_cb = OutfeedCallback(multi_activations_q,
                                     name="Multi-layer activations callback")

    callbacks = [gradients_cb]

    model = None
    if not no_pipelining:
        if model_type == REGULAR_MODEL:
            model = create_pipeline_model(
                multi_activations_q, gradient_accumulation_steps_per_replica
            )
        elif model_type == SEQUENTIAL_MODEL:
            model = create_pipeline_sequential_model(
                multi_activations_q, gradient_accumulation_steps_per_replica
            )
        callbacks += [multi_layer_cb]
    else:
        if not use_gradient_accumulation:
            gradient_accumulation_steps_per_replica = 1

        if model_type == SEQUENTIAL_MODEL:
            model = create_sequential_model(
                activations_q, gradient_accumulation_steps_per_replica
            )
        elif model_type == REGULAR_MODEL:
            model = create_model(activations_q,
                                 gradient_accumulation_steps_per_replica)
        callbacks += [layer_cb]

    if not model:
        raise Exception("Please select proper model_type!")

    return model, callbacks, optimizer_q

cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = num_ipus
cfg.configure_ipu_system()

strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
    model, callbacks, optimizer_outfeed_queue = \
        instantiate_selected_model_type(
            gradient_accumulation_steps_per_replica=
            gradient_accumulation_steps_per_replica,
            no_pipelining=no_pipelining,
            use_gradient_accumulation=use_gradient_accumulation,
            model_type=model_type,
            gradients_filters=gradients_filters,
            activations_filters=activations_filters
        )

    # Build the graph passing an OutfeedOptimizer to enqueue selected gradients
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=OutfeedOptimizer(
            wrapped_optimizer=keras.optimizers.SGD(),
            outfeed_queue=optimizer_outfeed_queue,
            outfeed_optimizer_mode=outfeed_optimizer_mode,
            model=model
        ),
        steps_per_execution=steps_per_epoch
    )

    # Train the model passing the callbacks to see the gradients
    # and activations stats
    model.fit(
        create_dataset(),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs
    )
