from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import ast

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
    # if type(tokenizer).__name__ == 'GPT2Tokenizer':
    #     ATTR_TO_SPECIAL_TOKEN['pad_token'] = '<pad>'
    #     print('<pad> token added!')
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    num_added_tokens = len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens) #50265, 4
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def preprocess_examples(examples):
    query = SPECIAL_TOKENS_MAP["query"] +" "+ str(examples["query"])
    persona = SPECIAL_TOKENS_MAP["persona"] + " "+ " </s> ".join(ast.literal_eval(examples['ground_persona']))
    knowledge = SPECIAL_TOKENS_MAP["knowledge"] + " " + str(examples["ground_knowledge"])
    answer = examples['answer']  
    text =  query+ persona + knowledge
    
    model_inputs = tokenizer.encode_plus(text, 
                                    max_length= 1024,
                                    padding = 'max_length',
                                    add_special_tokens=True,
                                    truncation=True,
                                    )
    labels = tokenizer.encode_plus(answer, 
                                            max_length= 256,
                                            padding = 'max_length',
                                            add_special_tokens=True,
                                            truncation=True,
                                            ).input_ids
    model_inputs["labels_raw"] = labels
    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

if __name__ == "__main__":
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")
    
    add_special_tokens_(model, tokenizer)
    df = pd.read_csv("../../data/focus_val_data.csv")
    row = df.iloc[5]
    model_inputs = preprocess_examples(row)
    # print(model_inputs['input_ids'])
    input_text = [tokenizer.decode(g, 
                            #   skip_special_tokens=True, 
                              clean_up_tokenization_spaces=True
                              ) for g in model_inputs['input_ids']]
    output_text = [tokenizer.decode(g, 
                            # skip_special_tokens=True, 
                          clean_up_tokenization_spaces=True
                            ) for g in model_inputs['labels_raw']]
    # tokenizer.decode(model_inputs['input_ids'])
    print("Decoded:")
    print("Input: ", input_text)
    print("Output: ", output_text)
    