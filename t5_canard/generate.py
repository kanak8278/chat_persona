import torch
import pandas as pd
import numpy as np
import ast
from pprint import pprint
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

def generate_utils(model, tokenizer, history, query, device):  
    history = str(" ||| ".join(ast.literal_eval(history)))
    text = history + " ||| " + query
    input_ids =  tokenizer(text, return_tensors = 'pt')
    input_ids = input_ids.to(device)
    generated_ids = model.generate(input_ids = input_ids['input_ids'], attention_mask = input_ids['attention_mask'])
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids][0]
    return preds, text

def generate(df):
    question_rewritten = []
    texts = [] 
    result = {}
    tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
    model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    for index, row in df.iterrows():
        if index %100 == 0:
            print(index)
        new_question, text = generate_utils(model, tokenizer, row['dialog_history'], row['query'], device)
        
        question_rewritten.append(new_question)
        texts.append(text)
    
    score = bert_score(question_rewritten, texts)
    result['bert_score'] = {"f1": sum(score['f1'])/len(score['f1']), 
                            "recall": sum(score['recall'])/len(score['recall']),
                            "precision": sum(score['precision'])/len(score['precision'])}
    
    df['question_rewritten'] = question_rewritten
    result['df'] = df
    return result


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")

# train_loc = "../data/focus_train_data.csv"
# val_loc = "../data/focus_val_data.csv"
# test_loc = "../data/focus_test_data.csv"

# train_df = pd.read_csv(train_loc)
# val_df = pd.read_csv(val_loc)
# test_df = pd.read_csv(test_loc)


if __name__ == "__main__":
    # result = generate(train_df)
    # result['df'].to_csv("./train_question_rewritten_1.csv", index=False)
    # print("BERT Score: ", result['bert_score'])





    df = pd.read_csv("./test_question_rewritten_1.csv")
    
    # bertscore = bert_score(list(df['question_rewritten']), list(df['query']))
    # avg_bertscore = {metric: sum(bert_score[metric])/len(bertscore[metric]) for metric in bertscore.keys()}
    # print("Avg BERT Score: ", avg_bertscore)
    
    bleuscore = bleu_score(list(df['question_rewritten']), list(df['query']))
    print("BLEU Score: ", bleuscore)
    
    rougescore = rouge_score(list(df['question_rewritten']), list(df['query']))
    print(rougescore)
    # avg_rougescore = {metric: sum(rougescore[metric])/len(rougescore[metric]) for metric in rougescore.keys()}
    # print("Avg ROUGE Score: ", avg_rougescore)
    
    
    # for idx, row in df.iterrows():
        
    #     print(row['query'], row['question_rewritten'])
    #     # score = bert_score(row['query'], row['question_rewritten'][0])
        
    #     if idx >=20:
    #         break