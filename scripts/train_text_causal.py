import sys
import os
# append a new directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/verma.shi/LLM/LitArt/data_module')
sys.path.append('/home/verma.shi/LLM/LitArt/models')

import argparse
import time
import json
from datetime import date

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping

from data_module.data_preprocessor import TextPreprocessing
from data_module.dataset import TextSummaryDataset
from data_module.datamodule import TextDataModule

#Transformers
import transformers
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM , AutoTokenizer
from transformers import pipeline, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from lightning.pytorch.loggers import TensorBoardLogger

#Dataset
from datasets import load_dataset

#PEFT
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training


import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

# Define a function to print the number of trainable parameters in the given model
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param}")

def tokenize_input(df,tokenizer,tokenizer_chapter_max_length,tokenizer_summary_max_length):

    prompt_start = "Summarize the following : \n"
    prompt_end = "\n\nSummary:"

    prompt = [prompt_start + dialogue + prompt_end for dialogue in df["chapter"]]

    df["input_ids"] = tokenizer(prompt, max_length=tokenizer_chapter_max_length , padding="max_length" , truncation=True , return_tensors="pt").input_ids
    df["labels"] = tokenizer(df["summary_text"],max_length=tokenizer_summary_max_length , padding="max_length" , truncation=True , return_tensors="pt").input_ids

    return df

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    #LLM Model Details
    parser.add_argument('--model' ,
                        type=str, help='Model to fine tune')
    parser.add_argument('--tokenizer',
                        type=str,help='Tokenizer to use')

    #Data folder location
    parser.add_argument('--trainpath',
                    type=str,help='Location for training file')
    parser.add_argument('--testpath',
                    type=str,help='Location for testing file')
    parser.add_argument('--valpath',
                    type=str,help='Location for validation file')

    #Fine Tunining Parameters
    parser.add_argument('--batchsize',
                    type=int,help='Batchsize')
    parser.add_argument('--chapterlength',
                    type=int,help='Chapter Length')
    parser.add_argument('--summarylength',
                    type=int,help='Summary Length')
    parser.add_argument('--num_epochs',
                    type=int,help='Number of epochs')

    #Logging details
    parser.add_argument('--log_path',
                type=str,help='Path to save logs')

    #Model Cache dir
    parser.add_argument('--cache_dir',
                type=str,help="Cache directory location")

    args = parser.parse_args()

    #Loading the data
    train_path = args.trainpath
    test_path = args.testpath
    val_path = args.valpath

    #Loading the model and tokenizer
    base_model_name = args.model
    tokenizer_name = args.tokenizer
    cache_dir = args.cache_dir

    #Fine Tuning Parameters
    tokenizer_chapter_max_length = args.chapterlength
    tokenizer_summary_max_length = args.summarylength

    #Training Parameters
    batch_size = args.batchsize
    epochs = args.num_epochs
    log_path = args.log_path

    #Initializing the dataloaders
    today = date.today()

    text_data = load_dataset('csv', 
                    data_files={
                        'train': train_path,
                        'test': test_path, 
                        'val': val_path})

    #Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #4bit quantizaition - load_in_4bit is used to load models in 4-bit quantization 
    bnb_4bit_use_double_quant=True, #nested quantization technique for even greater memory efficiency without sacrificing performance. This technique has proven beneficial, especially when fine-tuning large models
    bnb_4bit_quant_type="nf4", #quantization type used is 4 bit Normal Float Quantization- The NF4 data type is designed for weights initialized using a normal distribution
    bnb_4bit_compute_dtype=torch.bfloat16, #modify the data type used during computation. This can result in speed improvements. 
    )

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                      device_map="auto",
                                                      trust_remote_code=True,
                                                      quantization_config=bnb_config,
                                                      cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,cache_dir=cache_dir)
    # Set the padding token of the tokenizer to its end-of-sentence token
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing for the model. Gradient checkpointing is a technique used to reduce the memory consumption during the backward pas. Instead of storing all intermediate activations in the forward pass (which is what's typically done to compute gradients in the backward pass), gradient checkpointing stores only a subset of them
    base_model.gradient_checkpointing_enable() 

    # Prepare the model for k-bit training . Applies some preprocessing to the model to prepare it for training.
    base_model = prepare_model_for_kbit_training(base_model)


    # Define a configuration for LoRA (Low-Rank Adaptation). To create a LoRA model from a pretrained transformer model we use LoraConfig from PFET 
    config = LoraConfig(
        r=16, #The rank of decomposition r is << min(d,k). The default of r is 8.
        lora_alpha=32,#∆W is scaled by α/r where α is a constant. When optimizing with Adam, tuning α is similar as tuning the learning rate.
        target_modules=["query_key_value"], #Modules to Apply LoRA to target_modules. You can select specific modules to fine-tune.
        lora_dropout=0.05,#Dropout Probability for LoRA Layers #to reduce overfitting
        bias="none", #Bias Type for Lora. Bias can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. 
        task_type= "CAUSAL_LM", #Task Type
        )

    
    # Obtain a version of the model optimized for performance using the given LORA configuration
    base_model = get_peft_model(base_model, config)

    # Print the number of trainable parameters in the model
    print_trainable_parameters(base_model)



    print("Working")

    #Simplified Log 
    # log_path = log_path+base_model_name.replace("/","-")+"-" +str(today)+"-"+time.strftime("%H:%M:%S", time.localtime())
    # logger = TensorBoardLogger(log_path, name="my_model")
    
    # run_config = {
    #     "train_path":args.trainpath,
    #     "test_path":args.testpath,
    #     "val_path":args.valpath,
    #     "base_model_name":args.model,
    #     "tokenizer_name":args.tokenizer,
    #     "cache_dir":args.cache_dir,
    #     "batch_size":args.batchsize,
    #     "tokenizer_chapter_max_length":args.chapterlength,
    #     "tokenizer_summary_max_length":args.summarylength,
    #     "batch_size":args.batchsize,
    #     "epochs":args.num_epochs,
    #     "log_path":args.log_path,
    # }



    # # File path
    # filname = 'run_config.json'
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)

    # # Write the dictionary to a file in JSON format
    # with open(log_path+"/"+filname, 'w+') as file:
    #     json.dump(run_config, file)


