# Fine-tuning a pretrained transformer
This tutorial demonstrates how to fine-tune a pretrained model from the 
transformers library using IPUs. It is based on [Fine-tuning a pretrained model](https://huggingface.co/transformers/training.html).

## Environment preparation
Install the Poplar SDK following the instructions in the [Getting Started](https://docs.graphcore.ai/en/latest/software.html#getting-started)
guide for your IPU system. Make sure to run the enable.sh scripts for Poplar 
and PopART and activate a Python3 virtualenv with PopTorch installed.

Then install the package requirements:
```bash
pip install -r requirements.txt 
```

## Preparing the datasets

We use the IMDB dataset as our data. It contains movie reviews together with 
information on whether the review is positive or negative. To load the data we 
use the datasets library.


```python
from datasets import load_dataset

raw_datasets = load_dataset("imdb")
```


```python
raw_datasets.keys()
```




    dict_keys(['train', 'test', 'unsupervised'])



The `load_datasets` method returns a dictionary containing a dataset which is 
already split. We use the `train` split for training and the `test` split for 
validation.

The text must be transformed into a form understandable by the model. 
For this purpose, we create a function responsible for tokenization, which 
takes as input a batch from the dataset and returns a tokenized examples.
Note, that we set `max_length` and `truncation` parameters, which ensures 
that all examples have the same length. More detailed description can be found
in the [preprocessing data](https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation). 
Moreover, we remove the `text` field, because it is not accepted as input to 
the model.

We used ELECTRA as our model. It is an extension of BERT which 
is characterised by a shorter training time and therefore fits well into the 
tutorial. The model description together with implementation details can be 
found in [HuggingFace documentation](https://huggingface.co/transformers/model_doc/electra.html).


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/electra-small-generator")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function,
                                      remove_columns=['text'])
```

When the data has been processed, we can adjust it to the input of our model. 
To do this, we rename the column and set the format to `torch`, which will make 
the data to be stored in tensors.


```python
tokenized_datasets.set_format(type='torch')

train_dataset = tokenized_datasets["train"].rename_column(
    original_column_name='label', new_column_name='labels')
eval_dataset = tokenized_datasets["test"].rename_column(
    original_column_name='label', new_column_name='labels')
```

Next, when the datasets are ready we proceed to create the dataloaders. 
This is the first time we will use the capabilities of IPU - instead of using 
the `DataLoader` class from Pytorch we will use the implementation from 
PopTorch, which inherits from Pytorch and is optimised for memory usage and 
performance.

The data loading and the execution of the model on the IPU can be controlled 
using `poptorch.Options`. These options are used by PopTorch's wrappers 
such as `poptorch.DataLoader` and `poptorch.trainingModel`.


```python
import poptorch

opts = poptorch.Options().deviceIterations(8)
train_dataloader = poptorch.DataLoader(
    options=opts, dataset=train_dataset, shuffle=True, batch_size=4,
    drop_last=True
)

val_opts = poptorch.Options().deviceIterations(8) \
    .anchorMode(poptorch.AnchorMode.All)

eval_dataloader = poptorch.DataLoader(
    options=val_opts, dataset=eval_dataset, shuffle=True, batch_size=4,
    drop_last=True
)
```

In this example, we have configured `deviceIteration` and `anchorMode`.

**Device iteration** is one cycle of loop, which runs entirely on the IPU 
(the device), and which starts with a new batch of data. This option specifies 
the number of batches that are processed by the IPU during this cycle. 
The higher this number, the less the IPU has to interact with the CPU, 
for example, to request and wait for data, so that the IPU can loop faster. 
However, the user will have to wait for the IPU to go over all the iterations 
before getting the results back. 

**Anchor mode** specifies which data is returned from the model located on the 
IPU to the CPU. By default, PopTorch returns the last batch to the 
host machine after all iterations of the device, which is represented by
`AnchorMode.Final`. We set this parameter to `AnchorMode.All` to obtain every 
output from the model during the validation stage. This has an impact on the 
performance, due to the overhead of transferring more data to the host machine.

The full list of options is available in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#options).

## Preparing the model

Next, load the pretrained model and initialize the optimizer. Note that 
we use `poptorch.optim.AdamW`, which is optimised for distributed training.
More optimizers can be found [here](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#optimizers).


```python
from transformers import AutoModelForSequenceClassification
from poptorch.optim import AdamW

model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-small-generator",
    num_labels=2,
    return_dict=False,
    torchscript=True
)
optimizer = AdamW(model.parameters(), lr=1e-6)
```

Let's define how to split the model between the IPU devices. We use 4 
devices, and we place the embedding layer on the first device `IPU:0`. 
The encoder in the ELECTRA model consists of 12 layers, which we distribute 
equally with 4 layers on each device on each of 3 devices, finally we place 
the classifier layer on the last IPU:3 device.

In order to place a given layer on a particular device, we wrap it using 
`poptorch.BeginBlock()`, which takes as arguments the instance of 
`torch.nn.Module`, the name of the layer for display in the [PopVision profiler](https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/), 
and the device ID on which the layer should be placed.


```python
model.electra.embeddings = poptorch.BeginBlock(
    model.electra.embeddings, "Embedding", ipu_id=0
)
ipu_ids = [1] * 4 + [2] * 4 + [3] * 4
for index, (layer, ipu_id) in enumerate(zip(model.electra.encoder.layer, ipu_ids)):
    model.electra.encoder.layer[index] = poptorch.BeginBlock(
        layer, f"Encoder{index}", ipu_id=ipu_id
    )

model.classifier = poptorch.BeginBlock(
    model.classifier, "Classifier", ipu_id=3
)
```

We need to take one more step in order to adapt the model from HuggingFace to 
work with IPU. When a model uses multiple loss functions or uses a custom loss 
function, it has to be wrapped in `poptorch.identity_loss(loss)`.

Due to the fact that we can not directly modify the model class, we create 
a class that takes our ELECTRA model as a parameter and overload the `forward` 
function, in which we call the `forward` function from ELECTRA and then wrap 
the returned loss in `identity_loss`. Here we use the composition, however, 
this task could be solved using inheritance from the Electra class in 
transformers package.


```python
import torch


class IPUModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        loss, logits = self.model.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if self.model.training:
            loss = poptorch.identity_loss(loss, reduction="none")

        return loss, logits
```

Now, we create `trainingModel` and `inferenceModel`, for this task we use the 
`poptorch.trainingModel` and `poptorch.inferenceModel`. They take an instance 
of a `torch.nn.Module`, such as our model, an instance of `poptorch.Options` 
which we have instantiated previously, and an optimizer in case of 
`poptorch.trainingModel`. This wrapper uses TorchScript, and manages the 
translation of our model to a program that can be run using IPU. Then we will 
compile the models using one batch from our dataset.

In addition, the compilation of the model automatically attaches it to the 
device. We would like to use only one model (for the training or the inference) 
on the device at a time, as this will allow us to use a larger model. Therefore, 
we call `detachFromDevice` to detach the model from the IPU device.

Compilation may take a few minutes when using 4 devices. More information about 
the wrapping function can be found in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#model-wrapping-functions).


```python
trainingModel = poptorch.trainingModel(
    IPUModel(model), options=opts, optimizer=optimizer
)
trainingModel.compile(**next(iter(train_dataloader)))
trainingModel.detachFromDevice()
```


```python
inferenceModel = poptorch.inferenceModel(
    IPUModel(model), options=val_opts
)
inferenceModel.compile(**next(iter(eval_dataloader)))
inferenceModel.detachFromDevice()
```

## Training and validating the model

Our model and data are ready to run on IPU. We proceed to implement the 
function responsible for training the model. Here we set the model into the 
training state, and add a progress bar. We do not need to include 
`loss.backward()` as `poptorch.trainingModel` does this itself. 

At the beginning of our method, we attach the model to the IPU using the 
`attachToDevice` method, and at the end we detach it using `detachFromDevice`. 
It is worth noting here, that detaching the model from the device will 
automatically synchronise the updated weights of the model and transfer them 
to the CPU, so when we attach the model for the inference to the IPU device, 
the model will have the current state and copying the weights between devices 
is not necessary.


```python
from tqdm.auto import tqdm

epochs = 3
number_of_iterations = epochs * (len(train_dataloader) + len(eval_dataloader))
progress_bar = tqdm(range(number_of_iterations))


def train_epoch():
    trainingModel.train()
    trainingModel.attachToDevice()

    for batch in train_dataloader:
        loss, logits = trainingModel(**batch)
        progress_bar.update(1)

    trainingModel.detachFromDevice()
```

The function to validate is marked with the decorator `torch.no_grad()`, 
causes the gradients are not calculated. Moreover, in addition to setting the 
model into the evaluation state, we add storing predictions to count the 
accuracy at the end of the epoch.


```python
from sklearn.metrics import accuracy_score


@torch.no_grad()
def val_epoch():
    inferenceModel.eval()
    inferenceModel.attachToDevice()

    y_pred, y_true = [], []
    for batch in eval_dataloader:
        loss, logits = inferenceModel(**batch)

        y_pred += logits.argmax(dim=1).tolist()
        y_true.extend(batch['labels'].tolist())
        progress_bar.update(1)

    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc:.3f}')

    inferenceModel.detachFromDevice()
```

Finally, by bringing together everything we have written so far, we can start 
the training process.


```python
for epoch in range(epochs):
    train_epoch()
    val_epoch()
```

    Accuracy: 0.897


    Accuracy: 0.907


    Accuracy: 0.915


To sum up, in this tutorial, we have successfully fine-tuned a model from 
HuggingFace for sentiment prediction using IPU devices. If you are interested 
in other tutorials you are encouraged to check out [Graphcore Tutorials](https://github.com/graphcore/tutorials),
and for more advanced example can be found in [BERT example](https://github.com/graphcore/examples/tree/master/applications/pytorch/bert).
