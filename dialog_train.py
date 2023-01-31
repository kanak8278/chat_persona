import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import ast

val = "/content/drive/MyDrive/UMBC/data/focus_val_data.csv" #location to data file


# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

model.resize_token_embeddings(len(tokenizer))

# Define the dataset and dataloader for training
class DialogDataset(Dataset):
    def __init__(self, df, max_source_length = 512, max_target_length = 128, dialog_history = False):
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.dialog_history = dialog_history
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        row = self.df.iloc[item]
        query = str(row["query"])
        persona = str(" ".join(ast.literal_eval(row['ground_persona'])))
        context = str(row["ground_knowledge"])
        answer = str(row["answer"])
        
        
        if self.dialog_history:
          dialog_history = str(row["dialog_history"])
          text = dialog_history + "</s>" + query + "</s>" +  persona + "</s>" + context
        else:
          text = query + "</s>" + persona + "</s>" + context

        encoding = self.tokenizer.encode_plus(text, 
                                              max_length=self.max_source_length,
                                              padding = 'max_length',
                                              add_special_tokens=True,
                                              truncation=True,
                                              return_tensors="pt")
        answer_encoding = self.tokenizer.encode_plus(answer, 
                                                     max_length=self.max_target_length,
                                                     padding = 'max_length',
                                                     add_special_tokens=True,
                                                     truncation=True,
                                                     return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = answer_encoding["input_ids"]
        labels[labels == 0] = -100
        target_ids = labels.squeeze()
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        return input_ids, attention_mask, target_ids

# Define the optimizer and scheduler
optimizer = Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# Load the training data and split it into training and validation sets
df = pd.read_csv(val)
n_train = int(0.8 * len(df))
n_val = len(df) - n_train
train_df, val_df = df[:n_train], df[n_train:]

# Create the dataloaders
train_dataloader = DataLoader(DialogDataset(train_df), batch_size=2, shuffle=True)
val_dataloader = DataLoader(DialogDataset(val_df), batch_size=2, shuffle=False)

# Define the training loop
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=10, save_best=True):
    best_loss = float("inf")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i, (input_ids, attention_mask, target_ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = target_ids)
            loss = cross_entropy(output.view(-1, output.shape[-1]), target_ids.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(train_dataloader)
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, epochs, avg_loss))
        
        val_loss = 0
        with torch.no_grad():
            for i, (input_ids, attention_mask, target_ids) in enumerate(val_dataloader):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = cross_entropy(output.view(-1, output.shape[-1]), target_ids.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        print("Validation Loss: {:.4f}".format(avg_val_loss))
        
        if avg_val_loss < best_loss and save_best:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
        
        scheduler.step(avg_val_loss)

train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=10)

