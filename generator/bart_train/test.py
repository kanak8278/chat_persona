from models import *
from tqdm import tqdm
import json
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.bert import BERTScore
# from nubia_score import Nubia

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
# from bert_score import BERTScorer 
# from rouge import Rouge
# cc = SmoothingFunction()
# scorer = BERTScorer(lang="en", rescale_with_baseline=True)

from sys import exit
from pprint import pprint

# declaring a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def generate(model, tokenizer, knowledge, question, persona, device):
    input_ids, attention_mask = tokenizer(knowledge=knowledge, question=question, persona=persona)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    with torch.no_grad():
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=100, early_stopping=True)
        predictions = [predictions[0].tolist()]
        predictions = tokenizer.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return predictions

if __name__ == '__main__':
    
    # print("===================================================================================")
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/focus_test_data.csv")
    # print(df.columns)
    # pprint(df.iloc[12].to_dict())
    # print("===================================================================================")
    
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")
    # print(df.columns)
    # pprint(df.iloc[12].to_dict())
    
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")
    # df_1 = pd.read_csv("/home/ubuntu/chat_persona/generator/bart_train/neural_predictions_q_rewritten_1.csv")
    # df_0 = pd.read_csv("/home/ubuntu/chat_persona/generator/bart_train/neural_predictions_q_rewritten_0.csv")
    # print(df.columns)
    # print(df.shape, df_0.shape, df_1.shape)
    
    # for idx, row in enumerate(zip(df.iterrows(), df_0.iterrows(), df_1.iterrows())):
    #     if idx >= 200:   
    #         row, row_0, row_1 = row[0][1], row[1][1], row[2][1]
    #         print("Query:", row['query'])
    #         print("Query_Rewritten:", row_0['query'])
    #         persona = " ".join(ast.literal_eval(row['ground_persona']))
    #         print("Persona:", persona)
    #         print("Hit Knowledge:", row_0['knowledge'])
    #         print("Prediction_0 ", row_0['prediction'])
    #         print("Prediction_1: ", row_1['prediction'])
    #         print("Ground Answer: ", row['answer'])
    #         print("===============================================================")
    #     if idx >= 500:
    #         break
    
    

    

    # Generation Code
    
    # data = pd.read_csv("/home/ubuntu/chat_persona/data/knowledge_retrieval/test_query_rewritten_1_1.csv")
    # print(data.columns)
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")
    # print(df.columns)
    # print(df.shape, data.shape )
    
    # merge_df = pd.merge(data, df, on=['dialogID', 'utterance'], how='left')
    # print(merge_df.shape)
    # print(merge_df.columns)    
    # print("===================================================================================")
    # args = open('config.json').read()
    # args = json.loads(args)
    # args = dict2obj(args)
    # checkpoint_path = "/home/ubuntu/chat_persona/generator/bart_train/saved_weights/checkpoint-epoch=03-val_loss=0.32.ckpt"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer = Tokenizer(args)
    # model = FocusModel.load_from_checkpoint(checkpoint_path, args=args).eval().to(device)
    # # data = pd.read_csv("/home/ubuntu/chat_persona/data/knowledge_retrieval/test_query_rewritten_1_1.csv")

    # outputs = []
    # idx = 0
    # for row in tqdm(list(merge_df.iterrows())):
    #     idx+=1

    #     knowledge, question, persona, answer = row[1]['hit_knowledge_x'], row[1]['query_x'], row[1]['ground_persona'], row[1]['answer']
    #     persona = " ".join(ast.literal_eval(persona))
    #     if persona is None:
    #       persona = " "
    #     predictions = generate(model = model.generator, tokenizer = tokenizer, knowledge = knowledge, question = question, persona=persona, device=device)
    #     outputs.append([row[1]['query_x'], row[1]['hit_knowledge_x'], predictions[0] if type(predictions) is list else predictions])
        
    #     if idx%53==0:
    #         print("Query>>>>", question)
    #         print("Knowledge>>>>", knowledge)
    #         print("Ground>>>>", answer)
    #         print("Prediction>>>>", predictions[0] if type(predictions) is list else predictions)
    #         print("===============================================================")
    #     # if idx >= 200:
    #     #     break
            
    # outputs = pd.DataFrame(data=outputs, columns=['query', 'hit_knowledge', 'prediction'])
    # outputs['answer'] = merge_df['answer']
    # outputs.to_csv('./bart_predictions_exp_2.csv', index=False)
    
    df = pd.read_csv('/home/ubuntu/chat_persona/generator/bart_train/neural_predictions_q_rewritten_0.csv')
    print(df.columns)
    
    ground = list(df['answer'])
    preds = list(df['prediction'])
    # nubia = Nubia()
    # for grd, pred in zip(ground, preds):
    #     print("Ground: ", grd)
    #     print("Pred: ", pred)
    #     score = nubia.score(pred, grd, verbose=True, get_features=True)
    #     print("Nubia Score: ", score)
    #     print("===============================================================")
    #     break
    
    print("Metrics evaluation starting:")
    print("===================================================================================")
    
    # print("Calculating Bleu Score")
    # bleu = bleu_score(preds, ground)
    # print("Bleu Score: ", bleu)
    # print("===================================================================================")
    
    print("Calculating Rouge Score")
    rouge = rouge_score(preds[:], ground[:])
    print("Rouge Score: ", rouge)
    print("===================================================================================")
    
    bertscore = BERTScore('bert-base-uncased')
    print("Calculating BERT Score")
    bert = bertscore(preds[:], ground[:])
    bert = {k: sum(vv)/len(vv) for k, vv in bert.items()}
    print("BERT Score: ", bert)
    print("===================================================================================")
   
   
   
    # scores = {}
    # for idx, (pred, grd) in enumerate(zip(preds, ground)):
    #     bert = bert_score([pred], [grd])    
    #     for k, v in bert.items():
    #         if k not in scores:
    #             scores[k] = []
    #         scores[k].append(v)
    #     # if idx%100==0:
    #     #     break
    # scores = {k: sum(vv)/len(vv) for k, vv in scores.items()}
    # print(scores)