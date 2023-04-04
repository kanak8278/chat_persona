# data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import time
from pprint import pprint
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

import torch
import numpy as np # linear algebra
import pandas as pd 
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, IntervalStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


model_checkpoint = "google/electra-base-discriminator"
batch_size = 4
dataset = load_dataset("kanak8278/focus_persona_selection")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint, num_labels=1)

persona_candidate_columns = ["persona1", "persona2", "persona3", "persona4", "persona5", "persona6"] #persona6 for none of these 
# persona_grounding_columns = ["persona_grounding1", "persona_grounding2", "persona_grounding3", "persona_grounding4", "persona_grounding5"] #persona_grounding6 for none of these 

def preprocess_function(examples):

  first_sentences = [[f"{query} {hit_knowledge}"]*6 for query, hit_knowledge in zip(examples['query'], examples['hit_knowledge'])]
  second_sentences = [[examples[persona_candidate_column][i] for persona_candidate_column in persona_candidate_columns]for i, _ in enumerate(examples['dialogID'])]
  first_sentences = sum(first_sentences, [])
  second_sentences = sum(second_sentences, [])

  tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
  return {k: [v[i:i+6] for i in range(0, len(v), 6)] for k, v in tokenized_examples.items()}

dataset_encoded = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

# from dataclasses import dataclass
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
# from typing import Optional, Union
# import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        labels = list(map(int, labels))
        # print(labels)
        batch["labels"] = torch.tensor(labels)
        # batch["labels"] = torch.tensor(labels, dtype=torch.float64)
        return batch
    
if __name__ == "__main__":
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned_focus_swag_single_persona",
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=800, # Evaluate every half epoch
        # evaluation_strategy = IntervalStrategy(),
        learning_rate=5e-7,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=16,
        save_total_limit = 1,
        push_to_hub=True,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    

