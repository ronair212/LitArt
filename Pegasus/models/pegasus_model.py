
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

def load_model_details_pegasus(path="/work/LitArt/verma/final_models/google-pegasus-xsum-2024-03-20-17:16:12"):

    with open(path+"run_config.json") as json_file:
        run_details = json.load(json_file)
    
    base_model_name = run_details["base_model_name"]
    tokenizer_name = run_details["tokenizer_name"]
    cache_dir = run_details["cache_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,cache_dir=cache_dir)

    checkpoint_location = path+"my_model/version_0/checkpoints/*.ckpt"
    best_checkpoint_location = glob.glob(checkpoint_location)[0]

    model = torch.load(f=best_checkpoint_location,map_location=device)
    keys_to_modify = list(model["state_dict"].keys())  # Create a copy of the keys
    for key in keys_to_modify:
        new_key = key[6:]
        model["state_dict"][new_key] = model["state_dict"][key]
        del model["state_dict"][key]

    summary_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=base_model_name,state_dict=model["state_dict"])

    run_details["best_model_path"] = best_checkpoint_location
    
    return summary_model,base_model,tokenizer,run_details
