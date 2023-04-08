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
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMultipleChoice


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

def preprocess_function(examples, return_tensors=None):
  for key in examples.keys():
    if examples[key] is not list():
      examples[key] = [examples[key]]

  first_sentences = [[f"{query} {hit_knowledge}"]*6 for query, hit_knowledge in zip(examples['query'], examples['hit_knowledge'])]
  second_sentences = [[examples[persona_candidate_column][i] for persona_candidate_column in persona_candidate_columns]for i, _ in enumerate(examples['dialogID'])]
  first_sentences = sum(first_sentences, [])
  second_sentences = sum(second_sentences, [])
  
  tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding=True, return_tensors=return_tensors)

  return {k: [v[i:i+6] for i in range(0, len(v), 6)] for k, v in tokenized_examples.items()}

# dataset_encoded = dataset.map(preprocess_function, batched=True)



if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("kanak8278/xlnet-base-cased-finetuned_focus_swag_single_persona-finetuned_focus_swag_single_persona")
  model = AutoModelForMultipleChoice.from_pretrained("kanak8278/xlnet-base-cased-finetuned_focus_swag_single_persona-finetuned_focus_swag_single_persona")

  persona_candidate_columns = ["persona1", "persona2", "persona3", "persona4", "persona5", "persona6"] #persona6 for none of these 
  dataset = load_dataset("kanak8278/focus_persona_selection")

  
  examples = dataset['test'][0]
  # if type(examples) is not list:
  #   examples = [examples]
  print(examples.keys())
  
  
  results = []
  for data in tqdm(dataset['test']):
    
    hit_knowledge = data['hit_knowledge']
    persona1 = data['persona1']
    persona2 = data['persona2']
    persona3 = data['persona3']
    persona4 = data['persona4']
    persona5 = data['persona5']
    persona6 = data['persona6']
    label = data['label']

    inputs = preprocess_function(data.copy(), 'pt')
    outputs = model(**{k: v[0].unsqueeze(0).cuda()  for k, v in inputs.items() if k != "token_type_ids"})
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    req_columns = ['dialogID', 'utterance', 'label','query']
    result = {key:data[key] for key in req_columns}
    result['true_persona'] = data[f"persona{data['label']+1}"]
    result['pred_persona'] = data[f"persona{predicted_class+1}"]
    result['pred_class'] = predicted_class
    results.append(result)

  result_df = pd.DataFrame(results)
  # pred_value_counts = result_df['pred_class'].value_counts()
  # label_value_counts = result_df['label'].value_counts()
  result_df.to_csv("test_persona_results.csv", index=False)