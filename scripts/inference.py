import sys
import os
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

import argparse
import time

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import pipeline, set_seed

import torch
from diffusers import StableDiffusionPipeline

from LLama2.scripts.inference import generate_response
from utilities.helper_functions import text_to_prompt 

def summarize(chapter:str='',model_name:str= "google/pegasus-xsum", cache_dir: str = "/work/LitArt/cache")->str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)

    pipe = pipeline("summarization", model=model_pegasus ,tokenizer=tokenizer)

    pipe_out = pipe(chapter)[0]["summary_text"]
    
    return pipe_out

def generate(lora:str,prompt:str='',model_name:str= "CompVis/stable-diffusion-v1-4",file_name:str='test',inference_steps:int=32)->None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hashing = time.time()
    
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    print("--------------")
    print("Device:",device)
    print("-------------")
    pipe.load_lora_weights(lora,weight_name="pytorch_lora_weights.safetensors")
    image = pipe(prompt,
                negative_prompt="B&w,cropping,open book,no edges, cropped book, small book, other objects,square,edges clipping",
                guidance_scale=6.5,num_inference_steps=inference_steps).images[0]  

    image.save(f"/work/LitArt/sample_outputs/{file_name}_{hashing}.png")
    return image


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--chapter' ,
                        type=str, help='path to chapter to summarize')
    parser.add_argument('-s', '--summarizer',
                        default="google/pegasus-xsum",
                        type=str,help='name of model used for summarization')
    parser.add_argument('-g','--generator',
                        default="CompVis/stable-diffusion-v1-4",
                        type=str,help='name of model used for generation')
    parser.add_argument('-l', "--lora",
                        default="",
                        type=str,help='lora weights for trained model')
    
    parser.add_argument('-i',"--inference_steps",
                        default=32,
                        type=int,help='Image model inference steps')

    parser.add_argument('-f','--filename',
                        default="default",
                        type=str,help='name with which image is saved')
    
    parser.add_argument('-max_new_tokens','--max_new_tokens',
                        type=int,help='name with which image is saved')
    parser.add_argument('-do_sample','--do_sample',
                        type=str,help='name with which image is saved')
    parser.add_argument('-temperature','--temperature',
                        type=float,help='name with which image is saved')
    parser.add_argument('-top_p','--top_p',
                        type=float,help='name with which image is saved')

   
    args = parser.parse_args()

    hashing = time.time()

    chapter_path = args.chapter

    with open(chapter_path,mode='r') as f:
        chapter_text = f.read()

    
    s_model = args.summarizer
    g_model = args.generator
    file_name = args.filename
    l_weights = args.lora
    i_steps = args.inference_steps
    
    max_new_tokens = args.max_new_tokens
    do_sample = args.do_sample
    temperature = args.temperature
    top_p = args.top_p
    

    print("Generating summary....")
    if s_model=='Llama':
        summary_text = generate_response(chapter=chapter_text,
                                        model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13",
                                        max_new_tokens = max_new_tokens,
                                        do_sample = do_sample,
                                        temperature = temperature,
                                        top_p = top_p,
                                        )
    else:
        summary_text = chapter_text
    print(f"Summary: {summary_text}")
    
    with open(os.path.expanduser('~')+"/LitArt/utilities/summaries/"+f"{file_name}_{hashing}.txt", 'w') as file:
        file.write(summary_text)

    prompt = text_to_prompt(text=summary_text,model=l_weights)
    print(f"Prompt: {prompt}")
    print("Generating Image....")
    book_cover = generate(prompt=prompt,
             model_name=g_model,
             file_name=file_name,
             lora=l_weights,
             inference_steps=i_steps)
    
    print("Image generated!")
    



