import sys
sys.path.insert(1,'/home/patil.adwa/LitArt/LLama2')
from peft import LoraConfig
from utils.parameters import r, lora_alpha, lora_dropout , attention_blocks_lora 

def get_lora_config():
    
    #If only targeting attention blocks of the model
    
    if attention_blocks_lora == "True":
        target_modules = ["q_proj", "v_proj"]
    else:
        #If targeting all linear layers
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    

    return LoraConfig(
        r=r, 
        lora_alpha=lora_alpha,
        target_modules=target_modules, 
        lora_dropout=lora_dropout,
        bias="none", 
        task_type="CAUSAL_LM",
    )