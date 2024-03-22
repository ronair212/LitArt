import sys
sys.path.insert(1,'/home/patil.adwa/LitArt/LLama2')

import torch
from configs.bnb_config import get_bnb_config
from configs.peft_config import get_peft_config
from utils.parameters import cache_dir
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
#PEFT
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training




def get_inference_model(model_dir):
    
    '''
    bnb_config = get_bnb_config()
    peft_config = get_peft_config(model_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    '''
    
    


    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #quantization_config = PeftConfig.from_pretrained(model_dir)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)


    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        
    )





    return model, tokenizer
