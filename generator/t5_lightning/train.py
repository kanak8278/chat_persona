import os
import ast
import time
import torch
import wandb
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup


wandb.login()
COLUMNS = ['query', 'answer', 'ground_knowledge', 'ground_persona']
SPECIAL_TOKENS = [
    # "<machine>", "<human>",
    "<persona>", "<knowledge>", "<query>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': [
    # '<machine>', '<human>',
    '<persona>', '<knowledge>', '<query>']}
SPECIAL_TOKENS_MAP = {
    # "machine": "<machine>",
    # "human": "<human>",
    "persona": "<persona>",
    "knowledge": "<knowledge>",
    "query": "<query>"
}

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    num_added_tokens = len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def ast_literal(example):
    return " </s> ".join(ast.literal_eval(example))

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
print("T5 Model and Tokenizer loaded and ready!")

print(f"Model Vocab Size: {model.config.vocab_size}",
          f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
add_special_tokens_(model, tokenizer)
print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")


val = "../../data/focus_val_data.csv"



# Load the training data and split it into training and validation sets
df = pd.read_csv(val)
df['ground_persona'] = df['ground_persona'].apply(ast_literal)
n_train = int(0.8 * len(df))
n_val = len(df) - n_train
train_df, val_df = df[:n_train], df[n_train:]
print(train_df.shape, val_df.shape)

class DialogDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_length = 1024,
                 max_target_length = 256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
      row = self.df.iloc[item]
      query = SPECIAL_TOKENS_MAP["query"] +" "+ str(row["query"])
      persona = SPECIAL_TOKENS_MAP["persona"] + " "+ str(row['ground_persona'])
      knowledge = SPECIAL_TOKENS_MAP["knowledge"] + " " + str(row["ground_knowledge"])
      answer = row['answer']

      text =  query + " " + persona + " " + knowledge
      
      source_encoding = tokenizer.encode_plus(text, 
                                    max_length=self.max_source_length,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    )
      target_encoding = tokenizer.encode_plus(answer, 
                                    max_length=self.max_target_length,
                                    padding = 'max_length',
                                    # return_attention_mask = True,
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    ).input_ids

      input_ids = source_encoding["input_ids"].flatten()
      attention_mask = source_encoding["attention_mask"].flatten()

      labels = target_encoding
      
      labels[labels == 0] = -100

      # labels_with_ignore_index = []
      # for labels_example in labels:
      #   labels_example = [label if label != 0 else -100 for label in labels_example]
      #   labels_with_ignore_index.append(labels_example)


      target_ids = labels.flatten()
      return {
          # "query": query,
          # "persona": persona,
          # "knowledge": knowledge,
          # "answer": answer,
          "input_ids":input_ids,
          "attention_mask":attention_mask,
          "labels": target_ids
          }


train_dataset = DialogDataset(train_df, tokenizer)
train_dataloader = DataLoader(train_dataset,  batch_size=2)

valid_dataset = DialogDataset(val_df, tokenizer)
valid_dataloader = DataLoader(valid_dataset,  batch_size=6)

class CodeT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.resize_token_embeddings(new_num_tokens=32103)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return valid_dataloader


model = CodeT5()



# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=5,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')
wandb_logger = WandbLogger(name='codet5-finetune-personalized-answer-generation', project='CodeT5')
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    every_n_train_steps = 100,
    auto_insert_metric_name= True,
    monitor="val_loss",
    mode="min",
    dirpath="./CodeT5/",
    filename="t5model-nohistory-{epoch:02d}-{val_loss:.2f}",
)


trainer = Trainer(
    fast_dev_run=True,
    auto_lr_find=True,
    gpus=1, 
    default_root_dir="./CodeT5/Checkpoints", 
    logger=wandb_logger, 
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback])
print("Trainer Ready, Tuning Starts!")
tuned = trainer.tune(model)
print("Tuner Results: ", tuned)
print("Training Starts!")
trainer.fit(model)