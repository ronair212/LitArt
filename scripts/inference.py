import sys
sys.path.insert(1, '/home/patil.adwa/LitArt/utilities')

import argparse
import os
import time

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import pipeline, set_seed

import torch
from diffusers import StableDiffusionPipeline

from helper_functions import text_to_prompt

def summarize(chapter:str='',model_name:str= "google/pegasus-xsum")->str:
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    pipe = pipeline("summarization", model=model_pegasus ,tokenizer=tokenizer)

    pipe_out = pipe(chapter)[0]["summary_text"]
    
    return pipe_out

def generate(prompt:str='',model_name:str= "CompVis/stable-diffusion-v1-4",file_name:str='test')->None:

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    print("--------------")
    print("Device:",device)
    print("-------------")
    image = pipe(prompt,
                negative_prompt="B&w,cropping,open book,no edges, cropped book, small book, other objects,square,edges clipping",
                guidance_scale=6.5,num_inference_steps=32).images[0]  

    image.save(f"../sample_output/{file_name}_{time.time()}.png")


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
    parser.add_argument('-f','--filename',
                        default="default",
                        type=str,help='name with which image is saved')

    args = parser.parse_args()

    chapter_path = args.chapter

    with open(chapter_path,mode='r') as f:
        chapter_text = f.read()

    
    s_model = args.summarizer
    g_model = args.generator
    file_name = args.filename

    print("Generating summary....")
    # summary_text = summarize(chapter=chapter_text,
    #                          model_name=s_model)

    print(f"Summary: {chapter_text}")

    prompt = text_to_prompt(text=chapter_text)
    print(f"Prompt: {prompt}")
    print("Generating Image....")
    generate(prompt=prompt,
             model_name=g_model,
             file_name=file_name)
    
    print("Image generated!")
    



