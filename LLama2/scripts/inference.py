import sys
import os
sys.path.insert(1,"/work/LitArt/patel/LitArt/LLama2")
from models.inference_model import get_inference_model
from utils.logger import save_generated_summary
from utils.helper import string_to_bool
#from utils.parameters import max_new_tokens, do_sample , temperature, top_p

def extract_clean_response(input_string):
    # Split the input string by the marker "### Assistant:"
    parts = input_string.split("### Assistant: ")
    
    if len(parts) > 1:
        # Extract the first portion after the marker
        response_part = parts[1].strip()
        clean_response = response_part.split("###")[0].strip()
        clean_response = clean_response.replace("</s>", "").strip()
        return clean_response
    else:
        # If the marker  is not found, return the specified message
        response = "<Error in response from LLM>\n\n" + str(input_string)
        return response



def generate_response(chapter : str,model_dir:str ,max_new_tokens:int ,do_sample:str ,temperature:float ,top_p :float ) -> str:
    do_sample = string_to_bool(do_sample)
    model, tokenizer = get_inference_model(model_dir)
    prompt =  f"""### USER: Summarize the following text : ' {chapter}' ### Assistant:  """.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    outputs = model.generate(inputs.input_ids, 
                             max_new_tokens=max_new_tokens, 
                             do_sample=do_sample,
                             temperature = temperature,
                            top_p = top_p,)
    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return extract_clean_response(output)

#model_dir = '/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13'

#summary_generated = generate_response(chapter , model_dir)
#print(summary_generated)
#save_generated_summary(summary_generated)

