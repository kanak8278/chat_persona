import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import wandb

from utils import DialogDataset, train, validate

# wandb.init(project="t5_training_ground_knowledge")

args = {
    # Initialize config
    "TRAIN_BATCH_SIZE" : 16,    # input batch size for training (default: 64)
    "VALID_BATCH_SIZE" : 16,    # input batch size for testing (default: 1000)
    "TRAIN_EPOCHS" : 40,        # number of epochs to train (default: 10)
    "VAL_EPOCHS" : 1, 
    "LEARNING_RATE" : 5e-5,    # learning rate (default: 0.01)
    "SEED" : 42,               # random seed (default: 42)
    "MAX_LEN" : 512,
    "ANSWER_LEN" : 250, 
    "MODEL_SAVE_DIR" : "t5_grd_knw_grd_persona/"

}

# wandb.config.update(args)
    
train_params = {
        'batch_size': args["TRAIN_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 4
        }

val_params = {
    'batch_size': args["VALID_BATCH_SIZE"],
    'shuffle': True,
    'num_workers': 4
    }


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(args["SEED"]) # pytorch random seed
np.random.seed(args["SEED"]) # numpy random seed
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")
    
    train_loc = "../../data/focus_train_data.csv" #location to data file
    val = "../../data/focus_val_data.csv" #location to data file
    test = "../../data/focus_test_data.csv"
    
    train_df = pd.read_csv(train_loc)
    val_df = pd.read_csv(val)
    test_df = pd.read_csv(test)
    
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("VAL Dataset: {}".format(val_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    
    
    # Create the dataloaders
    train_dataloader = DataLoader(DialogDataset(tokenizer, train_df, args["MAX_LEN"], args["ANSWER_LEN"]), **train_params)
    val_dataloader = DataLoader(DialogDataset(tokenizer, val_df, args["MAX_LEN"], args["ANSWER_LEN"]), **val_params)
    test_dataloader = DataLoader(DialogDataset(tokenizer, test_df, args["MAX_LEN"], args["ANSWER_LEN"]), **val_params)
    print("Train Mini-Batch: ", len(train_dataloader), "Val Mini-Batch: ", len(val_dataloader), "Test Mini-Batch: ", len(test_dataloader))
    
    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(params =  model.parameters(), lr=args["LEARNING_RATE"])
    # scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
    
    # wandb.watch(model, log="all")
    print("Starting Training!")
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, args["TRAIN_EPOCHS"], args["MODEL_SAVE_DIR"], save_best=False)
    print()
    print("Loading inference model")
    inference_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_DIR)
    print("Inference Model Ready")
    predictions, actuals = validate(tokenizer, inference_model, test_dataloader)
    print("Predictions Ready!")
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f'{config.MODEL_SAVE_DIR}/t5_predictions.csv', index=False)
    print('Output Files generated for review')