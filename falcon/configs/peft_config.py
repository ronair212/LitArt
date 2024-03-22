import sys
sys.path.insert(1,'/home/nair.ro/LitArt/falcon')

from peft import PeftConfig

def get_peft_config(model_dir):
    return PeftConfig.from_pretrained(model_dir)
