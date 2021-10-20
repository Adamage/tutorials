"""
© Copyright 2020, The Hugging Face Team, Licenced under the Apache License,
Version 2.0
"""
"""
# Fine-tuning a pretrained transformer
This tutorial demonstrates how to fine-tune a pretrained model from the 
transformers library using IPUs. The tutorial extends the HuggingFace tutorial [Fine-tuning a pretrained model](https://huggingface.co/transformers/training.html)
with IPU specific code.
"""
"""
### Environment preparation
Install the Poplar SDK following the instructions in the Getting Started guide 
for your IPU system. Make sure to run the enable.sh scripts for Poplar and 
PopART and activate a Python3 virtualenv with PopTorch installed.

Then install the package requirements:
```bash
pip install -r requirements.txt 
```
"""
"""
### Preparing the datasets

As our data, we use the IMDB dataset containing movie reviews together with 
information whether the review is positive or negative. To load the data we 
use the datasets library.
"""

from datasets import load_dataset

raw_datasets = load_dataset("imdb")
raw_datasets.keys()
"""
The `load_datasets` method returns a dictionary containing a dataset which is 
already split. We use the "train" split for training and the "test" split for 
validation.
"""
"""
Next, the text must be transformed into a form understandable by the model. 
For this purpose, we create a function responsible for tokenization, which 
takes as input a batch from the dataset and returns a tokenized example.
Note that we set the `max_length` and `truncation` parameters, this ensures 
that all examples have the same length. Also, we remove the `text` field as 
it is not accepted as input to the model.

We used ELECTRA as our transformer to train. It is an extension of BERT which 
is characterised by a shorter training time and therefore fits well into the 
tutorial. The model description together with implementation details can be 
found in (HuggingFace documentation)[https://huggingface.co/transformers/model_doc/electra.html].
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/electra-base-generator")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function,
                                      remove_columns=['text'])
"""
When the data has been processed, we can adjust it to the input of our model. 
To do this, we rename the column and set the format to `torch` which will make 
the data to be stored in tensors.
"""

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

train_dataset = train_dataset.rename_column(original_column_name='label',
                                            new_column_name='labels')
eval_dataset = eval_dataset.rename_column(original_column_name='label',
                                          new_column_name='labels')

train_dataset.set_format(type='torch')
eval_dataset.set_format(type='torch')

"""
Next, when the datasets are ready we proceed to create the dataloaders. 
This is the first time we will use the capabilities of IPU - instead of using 
the `DataLoader` class from PyTroch we will use the implementation from 
PopTorch, which inherits from it and is optimised for memory usage and 
performance.

The dataloading and the execution of the model on the IPU can be controlled 
using `poptorch.Options`. These options are used by PopTorch's wrappers 
such as `poptorch.DataLoader` and `poptorch.trainingModel`.
"""
import poptorch

opts = poptorch.Options() \
    .deviceIterations(8)
train_dataloader = poptorch.DataLoader(
    options=opts,
    dataset=train_dataset,
    shuffle=True,
    batch_size=4,
    drop_last=True
)

val_opts = poptorch.Options() \
    .deviceIterations(8) \
    .anchorMode(poptorch.AnchorMode.All)

eval_dataloader = poptorch.DataLoader(
    options=val_opts,
    dataset=eval_dataset,
    shuffle=True,
    batch_size=4,
    drop_last=True
)
"""
In this example, we have configured `deviceIteration` and `anchorMode`.

*Device iterations* is one cycle of that loop, which runs entirely on the IPU 
(the device), and which starts with a new batch of data. This option specifies 
the number of batches that is prepared by the host (CPU) for the IPU. 
The higher this number, the less the IPU has to interact with the CPU, 
for example to request and wait for data, so that the IPU can loop faster. 
However, the user will have to wait for the IPU to go over all the iterations 
before getting the results back. 

*Anchor mode* specifies which data is returned from the model located on the 
IPU to the CPU. By default, PPopTorch will only return the last batch to the 
host machine after all iterations of the device, which is represented by
`AnchorMode.Final`. We set this parameter to `AnchorMode.All` to obtain every
model output during the validation stage. This has an impact on the performance,
due to overhead of transferring more data to the host machine.

The list of these options is available in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#options).
"""
"""

"""
import torch


class Wrapped(torch.nn.Module):
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
            final_loss = poptorch.identity_loss(loss, reduction="none")
            return final_loss, logits

        return loss, logits


"""

"""
from transformers import AutoModelForSequenceClassification
from poptorch.optim import AdamW

model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-base-generator",
    num_labels=2,
    return_dict=False,
    torchscript=True
)
optimizer = AdamW(model.parameters(), lr=1e-5)

"""

"""

model.electra.embeddings = poptorch.BeginBlock(
    model.electra.embeddings, "Embedding", ipu_id=0
)

layer_ipu = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
for index, layer in enumerate(model.electra.encoder.layer):
    ipu = layer_ipu[index]
    model.electra.encoder.layer[index] = poptorch.BeginBlock(
        layer, f"Encoder{index}", ipu_id=ipu
    )

model.classifier = poptorch.BeginBlock(
    model.classifier, "Classifier", ipu_id=3
)
"""

"""

trainingModel = poptorch.trainingModel(Wrapped(model), options=opts)
trainingModel.compile(**next(iter(train_dataloader)))

inferenceModel = poptorch.inferenceModel(Wrapped(model), options=val_opts)
inferenceModel.compile(**next(iter(eval_dataloader)))

"""

"""
from tqdm.auto import tqdm

epochs = 2
number_of_iterations = epochs * (len(train_dataloader) + len(eval_dataloader))
progress_bar = tqdm(range(number_of_iterations))


def train_epoch():
    trainingModel.train()
    for batch in train_dataloader:
        loss, logits = trainingModel(**batch)
        train_loss = loss.item()
        progress_bar.update(1)


"""

"""
from sklearn.metrics import accuracy_score


def val_epoch():
    inferenceModel.eval()
    y_pred, y_true = [], []
    for batch in eval_dataloader:
        y_true.extend(batch['labels'].tolist())

        with torch.no_grad():
            loss, logits = inferenceModel(**batch)

        y_pred.extend(logits.argmax(dim=1).tolist())
        progress_bar.update(1)

    acc = accuracy_score(y_true, y_pred)
    print(f'{acc:.3f}')


"""

"""


def update_weights():
    trainingModel.copyWeightsToHost()
    inferenceModel.copyWeightsToDevice()


"""

"""
for epoch in range(epochs):
    train_epoch()
    update_weights()
    val_epoch()

"""

"""

train_dataloader.terminate()
eval_dataloader.terminate()

"""

"""
