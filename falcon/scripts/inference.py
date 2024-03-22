import sys
import os
sys.path.insert(1,"/home/nair.ro/LitArt/flacon")
from models.inference_model import get_inference_model
from utils.logger import save_generated_summary


def generate_response(chapter : str,model:str) -> str:
    #Inference
    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    
    prompt =  f'''
    "Summarize the following : \n" {chapter}
    "\n\nSummary:" 
    '''.strip()
    DEVICE  = "cuda"
    encoding = tokenizer(prompt, return_tensors = "pt").to(DEVICE)
    generation_config = generation_config()
    #with torch.inference_mode():
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    return response.strip()

#model_dir = '/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13'

#summary_generated = generate_response(chapter , model_dir)
#print(summary_generated)
#save_generated_summary(summary_generated)