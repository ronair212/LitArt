import sys
sys.path.insert(1,'/home/patil.adwa/LitArt/LLama2')

import torch
from transformers import BitsAndBytesConfig
from utils.parameters import quant_4bit, quant_8bit


def get_bnb_config():
    if quant_4bit == "True" :
        return BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            llm_int8_enable_fp32_cpu_offload=True
        )
    if quant_8bit == "True" :
        return BitsAndBytesConfig(
            load_in_8bit=True, 
            llm_int8_threshold=0.0,
        )

