from models import *
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
   
# declaringa a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

if __name__ == '__main__':
    args = open('config.json').read()
    args = json.loads(args)
    args = dict2obj(args)
    
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # tokenizer = GPT2Tokenizer.from_pretrained(
    #         "EleutherAI/gpt-neo-1.3B",    
    #         bos_token='<|startoftext|>',
    #         eos_token='<|endoftext|>',
    #         pad_token='<|pad|>')
    
    # """Debugging code"""
    # train_dataset = FocusDataset(args, train=True)
    # for idx, data in enumerate(train_dataset):
    #     (input_ids, attention_mask, persona), (labels, decoder_attention_mask) = data
    #     # print(persona)
    #     print([tokenizer.decode(input_id , skip_special_tokens=True) for input_id in input_ids])
    #     print()
    #     if idx >= 5:        
            # break



    """Training Code Starts here"""    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, strict=False, min_delta=0.001 , verbose=True, mode='min')
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath="./saved_weights", filename='bart_0_0_1-checkpoint-{epoch:02d}-{val_loss:.3f}', save_top_k=1, mode='min')
    
    trainer_args = {
       'accelerator': args.accelerator,
        'devices' : args.devices,
        'max_epochs' : args.max_epoch,
        'val_check_interval' : args.val_check_interval,
        'precision' : args.precision,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'fast_dev_run' : True,
        # 'strategy': args.strategy,
    }

    trainer = pl.Trainer(**trainer_args, callbacks=[early_stop_callback, model_checkpoint_callback])
    
    model = FocusModel(args)
    
    if args.load_from_checkpoint:
        model.load_from_checkpoint(checkpoint_path = args.checkpoint_path, args = args)
    
    dm = FocusDataModule(args)
    trainer.fit(model, dm) 