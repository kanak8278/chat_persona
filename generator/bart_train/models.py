import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from termcolor import colored
import textwrap
import ast
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    AdamW,
)

from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib import rcParams, rc

pl.seed_everything(42)

train_df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = BartTokenizer.from_pretrained(args.model_generator)
        self.max_len_context = args.max_len_context
        self.max_len_answer = args.max_len_answer
        self.max_len_question = args.max_len_question
        self.max_len_persona = args.max_len_persona
        self.task = args.task
        
    def __call__(self, knowledge=None, question=None, persona = None, answer=None,):
        if answer:
            if type(answer) is not list:
                answer = [answer]
            batch_answer = self.tokenizer(answer,
                                          max_length=self.max_len_answer,
                                          padding='max_length',
                                          truncation=True,
                                          add_special_tokens=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
            labels = batch_answer.input_ids.clone()
            labels[labels==0] = -100
            return labels, batch_answer.attention_mask

        else:
            if type(knowledge) is not list:
                knowledge = [knowledge]
            if type(question) is not list:
                question = [question]
            if type(persona) is not list:
                persona = [persona]
            
                
            # add separate token to answers
            knowledge = [self.task + ' ' + c for c in knowledge]
            persona = [' ' + str(per) for per in persona]
            question = [' ' + str(ques) for ques in question]
            
            batch_knowledge = self.tokenizer(knowledge,
                                             max_length=self.max_len_context,
                                             padding='max_length',
                                             truncation=True,
                                             add_special_tokens=True)
            batch_question = self.tokenizer(question,
                                            max_length=self.max_len_persona,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True)
            batch_persona = self.tokenizer(persona,
                                           max_length=self.max_len_question,
                                           padding='max_length',
                                           truncation=True,
                                           add_special_tokens=True)
            
            input_ids = torch.LongTensor([k[:-1] + p + q for (k, p, q) in zip(batch_knowledge.input_ids, batch_persona.input_ids, batch_question.input_ids)])
            attention_mask = torch.FloatTensor([k[:-1] + p + q for (k, p, q) in zip(batch_knowledge.attention_mask, batch_persona.attention_mask, batch_question.attention_mask)])
            # print(input_ids.shape)
            return input_ids, attention_mask
    
       
class FocusDataset(Dataset):
    def __init__(self, args, train=True):
        self.tokenizer = Tokenizer(args)
        if train:
            self.dataset = pd.read_csv(args.train_dataset)
        else:
            self.dataset = pd.read_csv(args.val_dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw = self.dataset.iloc[idx]
        knowledge, question, answer, persona = raw['ground_knowledge'], raw['question_rewritten'], raw['answer'], raw['ground_persona']
        persona = " ".join(ast.literal_eval(persona))
        if persona is None:
          persona = " "
        return self.tokenizer(knowledge=knowledge, question=question, persona=persona), self.tokenizer(answer=answer)


class FocusDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args 
        self.tokenizer = Tokenizer(args)
    
    def prepare_data(self):
        self.train_dataset = FocusDataset(self.args, train=True)
        self.test_dataset = FocusDataset(self.args, train=False)
        
    def setup(self, stage=None):
        val_length = int(self.args.val_test_split_factor * len(self.test_dataset))
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_length, len(self.test_dataset) - val_length])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=False)

class FocusModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BartForConditionalGeneration.from_pretrained(args.model_generator)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.automatic_optimization = False 
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler' : torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, gamma=self.args.lr_scheduler_factor),
            'interval' : 'step',
            'frequency' : 1,
            'strict' : True
        }
        return [optimizer], [scheduler]
    


    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=labels)
        # output = self.generator.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch[0][0].squeeze(1), batch[0][1].squeeze(1)
        labels, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
        loss, outputs = self(input_ids= input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch[0][0].squeeze(1), batch[0][1].squeeze(1)
        labels, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
        loss, outputs = self(input_ids= input_ids,
             attention_mask=attention_mask,
             decoder_attention_mask=decoder_attention_mask,
             labels=labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('ptl/val_loss', avg_loss)
    
    # def compute_loss(self, batch, batch_idx):
    #     input_ids, attention_mask = batch[0][0].squeeze(1), batch[0][1].squeeze(1)
    #     decoder_input_ids, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
    #     out = self.generator(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids)
    #     loss, logits = out.loss, out.logits
    #     prob = torch.functional.softmax(logits, dim=-1)
    #     batch_size = input_ids.shape[0]
    #     loss_tune = torch.zeros(batch_size)
        
    #     for i in range(batch_size):
    #         prediction = self.tokenizer.decode(prob[i, :, :].argmax(dim=-1).tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         ground_truth = self.tokenizer.decode(decoder_input_ids[i, :].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         persona = self.tokenizer.decode(input_ids[i, self.args.max_len_context-2:self.args.max_len_context+64].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         # print(persona)
    #         r = 1 - self.evaluator(prediction, ground_truth, persona)
    #         loss_tune[i] = r * self.cross_entropy_loss(logits[i], decoder_input_ids[i])
        
    #     loss_tune = loss_tune.mean()
    #     loss = loss * self.gamma + (1 - self.gamma) * loss_tune
    #     return loss
    
    # def validation_step(self, batch, batch_idx):
    #     loss = self.compute_loss(batch, batch_idx)
    #     self.log('val_loss', loss, on_step=True)
    #     return loss
    
    


if __name__ == '__main__':    
    persona_token_counts, knowledge_token_counts, question_token_counts, answer_token_counts = [], [], [], []


    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # for _, row in train_df.iterrows():
    #     persona_token_count = len(tokenizer.encode(row['ground_persona']))
    #     persona_token_counts.append(persona_token_count)
        
    #     knowledge_token_count = len(tokenizer.encode(row['ground_knowledge']))
    #     knowledge_token_counts.append(knowledge_token_count)
        
    #     question_token_count = len(tokenizer.encode(row['question_rewritten']))
    #     question_token_counts.append(question_token_count)
        
    #     answer_token_count = len(tokenizer.encode(row['answer']))
    #     answer_token_counts.append(answer_token_count)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    # sns.histplot(persona_token_counts, ax=ax1)
    # ax1.set_title('Persona')

    # sns.histplot(knowledge_token_counts, ax=ax2)
    # ax2.set_title('Knowledge')

    # sns.histplot(question_token_counts, ax=ax3)
    # ax3.set_title('Question')

    # sns.histplot(answer_token_counts, ax=ax4)
    # ax4.set_title('Answer')
    # plt.savefig('token_counts.png')
        
