import sys
sys.path.insert(1,'/home/nair.ro/LitArt/falcon')
from peft import LoraConfig
from utils.parameters import r, lora_alpha, lora_dropout

def get_lora_config():
    return LoraConfig(
        r=r, 
        lora_alpha=lora_alpha,
        target_modules=["query_key_value"], 
        lora_dropout=lora_dropout,
        bias="none", 
        task_type="CAUSAL_LM",
    )
