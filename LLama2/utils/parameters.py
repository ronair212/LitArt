import sys
import os
sys.path.insert(1,'/home/nair.ro/LitArt/LLama2')


from datetime import date
import time

# Define global parameters 
tokenizer_chapter_max_length = 1024
tokenizer_summary_max_length = 128
base_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer_name = "meta-llama/Llama-2-7b-hf"
cache_dir = "/work/LitArt/nair/cache/" 
log_path = "/work/LitArt/nair/outdir/"



# Training Parameters
batch_size = int(os.getenv('BATCH_SIZE'))
epochs = int(os.getenv('EPOCHS'))
gradient_accumulation_steps = 2
learning_rate = 1e-4
save_total_limit = 3
logging_steps = 10
max_steps = 200
today = date.today()
output_dir = log_path + base_model_name.replace("/", "-") + "-" + str(today) + "-" + time.strftime("%H:%M:%S", time.localtime())
save_strategy = 'epoch'

r = int(os.getenv('R', 16))
lora_alpha = int(os.getenv('LORA_ALPHA', 32))
lora_dropout = float(os.getenv('LORA_DROPOUT', 0.05))


quant_4bit = str(os.getenv('QUANT_4BIT'))
quant_8bit = str(os.getenv('QUANT_8BIT'))

attention_blocks_lora = str(os.getenv('ATTN_BLOCK_LORA'))