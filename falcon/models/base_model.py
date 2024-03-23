import sys
sys.path.insert(1,'/home/nair.ro/LitArt/falcon')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.bnb_config import get_bnb_config
from utils.parameters import base_model_name, tokenizer_name, cache_dir
from peft import prepare_model_for_kbit_training
from utils.logger import print_trainable_parameters

def get_base_model():
    bnb_config = get_bnb_config()
    lora_config = get_lora_config()

    '''
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                      device_map="auto",
                                                      trust_remote_code=True,
                                                      quantization_config=bnb_config,
                                                      cache_dir=cache_dir)
                                                      
    '''

    base_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                      device_map="auto",
                                                      trust_remote_code=True, 
                                                      quantization_config=bnb_config , 
                                                      cache_dir=cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.add_adapter(lora_config)
    
    ## log the number of trainable parameters in the model 
    print_trainable_parameters(base_model)
    
    return base_model, tokenizer
