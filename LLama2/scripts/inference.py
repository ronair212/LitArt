import sys
import os
sys.path.insert(1,"/home/patil.adwa/LitArt/LLama2")
from models.inference_model import get_inference_model
from utils.logger import save_generated_summary

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
        return "No response from LLM"


def generate_response(chapter : str,model_dir:str) -> str:
    model, tokenizer = get_inference_model(model_dir)
    prompt =  f"""### USER: Summarize the following text : ' {chapter}' ### Assistant:  """.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    outputs = model.generate(inputs.input_ids, max_new_tokens=500, do_sample=False)
    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return extract_clean_response(output)


# summary_generated = generate_response(chapter)
# print(summary_generated)
# save_generated_summary(summary_generated)

