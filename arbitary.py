import torch
import pandas as pd
from pprint import pprint
import ast

# import log

if __name__ == "__main__":


    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    # # Let's chat for 5 lines
    # for step in range(5):
    #     # encode the new user input, add the eos_token and return a tensor in Pytorch
    #     new_user_input_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt')
        
    #     # append the new user input tokens to the chat history
    #     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    #     print(bot_input_ids)
    #     print(bot_input_ids.shape)
        
        
    #     # print(bot_input_ids.view(-1).shape)        
        
    #     decoded_text = tokenizer.batch_decode(bot_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     pprint(decoded_text)
    #     # generated a response while limiting the total chat history to 1000 tokens, 
    #     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    #     # pretty print last ouput tokens from bot
    #     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

    
    
    # tokenizer = T5Tokenizer.from_pretrained("t5-base", model)
    # print(tokenizer)
    df = pd.read_csv("/work/kanakr/chat_persona/data/focus_test_data.csv")
    print(df['query'].iloc[2])
    
    # dialog_history = dialog = ' EOS '.join(ast.literal_eval(df['dialog_history'].iloc[2]))
    # print(dialog_history)
    # test_data = DialogDataset(tokenizer, df, max_source_length = 512, dialog_history=True)
    
    # decoded_text = tokenizer.decode(test_data[5]['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(len(decoded_text))
    # print(decoded_text)
    