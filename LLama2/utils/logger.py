import sys
sys.path.insert(1,'/home/nair.ro/LitArt/LLama2')
import time
from datetime import date
import os
import transformers
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM , AutoTokenizer
from transformers import pipeline, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
#PEFT
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training

from utils.parameters import log_path
from utils.parameters import base_model_name
today = date.today()

#cd /work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13/
model = base_model_name
log_path = log_path+model.replace("/","-")+"-" +str(today)+"-"+time.strftime("%H:%M:%S", time.localtime())

from lightning.pytorch.loggers import TensorBoardLogger

def get_logger(log_path):
    return TensorBoardLogger(log_path, name="my_model")

def save_hyperparameters(log_path, quantization_config, lora_config , training_arguments):
        
        os.makedirs(log_path, exist_ok=True)
        
        file_path = os.path.join(log_path, 'hyperparameters.txt')  
        
        with open(file_path, 'w') as file:
            file.write(str(quantization_config))
            file.write(str(lora_config))
            file.write(str(training_arguments))
            
            
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
            
    #print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param}")
    #save to file
    
    os.makedirs(log_path, exist_ok=True)
    file_path = os.path.join(log_path, 'number_of_trainable_para.txt')  

    with open(file_path, 'w') as file:
        file.write(str(print_trainable_parameters(model)))
        
def save_generated_summary(summary_generated):
    file_path = os.path.join(log_path, 'summary.txt')  

    with open(file_path, 'w') as file:
        file.write(summary_generated)
