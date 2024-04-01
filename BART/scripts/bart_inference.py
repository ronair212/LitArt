import sys
import json
import glob
import tqdm
import pandas as pd
import torch
import evaluate
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
tqdm.tqdm.pandas()
from summarizer import TextSummaryModel
cache_dir="/work/LitArt/cache"
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

from T5.models import load_model_details_bart

def summarize_bart(text):

    summary_model,base_model,tokenizer,run_details = load_model_details_bart()
    
    chapter_length=run_details["tokenizer_chapter_max_length"]
    summary_length=run_details["tokenizer_summary_max_length"]
    temperature=1.5
    repetition_penalty=1.5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    text = "Summarize the following : \n" + text
    inputs = tokenizer(text, 
                       max_length=chapter_length,
                       truncation=True,
                       padding="max_length",
                       add_special_tokens=True, 
                       return_tensors="pt").to(device)
    summarized_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_length= summary_length,
            temperature = temperature,
            do_sample = True,
            repetition_penalty = repetition_penalty).to(device)

    return " ".join([tokenizer.decode(token_ids, skip_special_tokens=True)
                    for token_ids in summarized_ids])
    