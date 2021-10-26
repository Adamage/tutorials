# © Copyright 2021 Graphcore Ltd. All rights reserved.
# 
# © Copyright 2020, The Hugging Face Team, Licenced under the Apache License,Version 2.0
from datasets import load_dataset

raw_datasets = load_dataset("imdb")

raw_datasets.keys()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/electra-small-generator")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function,
                                      remove_columns=['text'])

tokenized_datasets.set_format(type='torch')

train_dataset = tokenized_datasets["train"].rename_column(
    original_column_name='label', new_column_name='labels')
eval_dataset = tokenized_datasets["test"].rename_column(
    original_column_name='label', new_column_name='labels')

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

from transformers import AutoModelForSequenceClassification
from poptorch.optim import AdamW

model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-small-generator",
    num_labels=2,
    return_dict=False,
    torchscript=True
)
optimizer = AdamW(model.parameters(), lr=1e-6)

model.electra.embeddings = poptorch.BeginBlock(
    model.electra.embeddings, "Embedding", ipu_id=0
)
ipu_ids = [1] * 4 + [2] * 4 + [3] * 4
for index, (layer, ipu_id) in enumerate(
        zip(model.electra.encoder.layer, ipu_ids)):
    model.electra.encoder.layer[index] = poptorch.BeginBlock(
        layer, f"Encoder{index}", ipu_id=ipu_id
    )

model.classifier = poptorch.BeginBlock(
    model.classifier, "Classifier", ipu_id=3
)

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

trainingModel = poptorch.trainingModel(
    IPUModel(model), options=opts, optimizer=optimizer
)
trainingModel.compile(**next(iter(train_dataloader)))
trainingModel.detachFromDevice()

inferenceModel = poptorch.inferenceModel(
    IPUModel(model), options=val_opts
)
inferenceModel.compile(**next(iter(eval_dataloader)))
inferenceModel.detachFromDevice()

from tqdm.auto import tqdm

epochs = 3
number_of_iterations = epochs * (len(train_dataloader) + len(eval_dataloader))
progress_bar = tqdm(range(number_of_iterations))


def train_epoch():
    trainingModel.train()

    for batch in train_dataloader:
        loss, logits = trainingModel(**batch)
        progress_bar.update(1)

    trainingModel.detachFromDevice()

from sklearn.metrics import accuracy_score


@torch.no_grad()
def val_epoch():
    inferenceModel.eval()

    y_pred, y_true = [], []
    for batch in eval_dataloader:
        loss, logits = inferenceModel(**batch)

        y_pred += logits.argmax(dim=1).tolist()
        y_true.extend(batch['labels'].tolist())
        progress_bar.update(1)

    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc:.3f}')

    inferenceModel.detachFromDevice()

for epoch in range(epochs):
    train_epoch()
    val_epoch()
