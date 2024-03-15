import os
print (os.environ['CONDA_DEFAULT_ENV'])


import transformers
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())


import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk
import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
from datasets import Dataset, DatasetDict
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch
from transformers import PegasusForConditionalGeneration


nltk.download("punkt")



data = load_dataset("kmfoda/booksum")

train_df = data['train'].to_pandas()
validation_df = data['validation'].to_pandas()
test_df = data['test'].to_pandas()
combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)




def pre_processing(sentence):
    #lower text
    sentence = sentence.lower()

    pattern = r"\s*\([a-zA-Z]\s_\)"
    sentence = re.sub(pattern, "", sentence)
        
    sentence = sentence.replace("\n"," ")
    
    # replacing everything with space  
    sentence = re.sub(r"[=.!,¿?.!+,;¿/:|%()<>।॰{}#_'\"@$^&*']", " ", sentence)
    sentence = re.sub(r"…", " ", sentence)
    
    #remove double quotes
    sentence = re.sub(r'"', " ", sentence)
    
    #remove numbers
    sentence = re.sub(r'[0-9]', "", sentence)
    #sentence = re.sub(r'#([^s]+)', r'1', sentence)
    
    #remove website links
    sentence = re.sub('((www.[^s]+)|(https?://[^s]+))','',sentence)
    
    #remove @anythin here
    #sentence = re.sub('@[^s]+','',sentence)
    
    #remove multiple spaces
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    # remove extra space
    sentence = sentence.strip()
    
    return sentence


required_columns = ['chapter', 'summary_text']

df_new = combined_df[required_columns]

df_new['summary_text'] = df_new['summary_text'].apply(lambda x: pre_processing(x))

df_new['chapter'] = df_new['chapter'].apply(lambda x: pre_processing(x))






train, test_val = train_test_split(df_new, test_size=0.2, random_state=42)

test, val = train_test_split(test_val, test_size=0.5, random_state=42)  # 0.25 x 0.8 = 0.2



train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)
val_dataset = Dataset.from_pandas(val)

Dataset_dic = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})


from datasets import Dataset, DatasetDict

def split_chapters_into_chunks(dataset_dict):
    def chunk_text(text, chunk_size=1000):
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])
    
    def process_dataset(dataset):
        new_rows = []
        for row in dataset:
            chapter_chunks = list(chunk_text(row['chapter'], 1000))
            for chunk in chapter_chunks:
                new_row = row.copy()
                new_row['chapter'] = chunk
                new_rows.append(new_row)
        return Dataset.from_pandas(pd.DataFrame(new_rows))
    
    # Process each dataset within the DatasetDict
    processed_datasets = {k: process_dataset(dataset_dict[k]) for k in dataset_dict.keys()}
    return DatasetDict(processed_datasets)

# Assume `dataset_dict` is your input DatasetDict with 'train', 'validation', and 'test' datasets
# You would call the function like this:
new_dataset_dict = split_chapters_into_chunks(Dataset_dic)




from datasets import Dataset, DatasetDict

def split_dataset_dict_equally(dataset_dict, num_parts=10):
    new_dataset_dicts = [{} for _ in range(num_parts)]

    # Function to calculate split sizes and handle remainders
    def calculate_split_sizes(total_size, num_parts):
        base_size = total_size // num_parts
        remainder = total_size % num_parts
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_parts)]
        return sizes

    # Split each dataset and distribute rows evenly across new dataset dicts
    for split_name, dataset in dataset_dict.items():
        total_size = len(dataset)
        split_sizes = calculate_split_sizes(total_size, num_parts)
        offsets = [sum(split_sizes[:i]) for i in range(num_parts)]
        
        for i, offset in enumerate(offsets):
            # For the last part, take all remaining rows to ensure no row is left behind
            if i == num_parts - 1:
                part_dataset = dataset.select(range(offset, total_size))
            else:
                part_dataset = dataset.select(range(offset, offset + split_sizes[i]))
            new_dataset_dicts[i][split_name] = part_dataset

    # Wrap each dict in a DatasetDict
    new_dataset_dicts = [DatasetDict(part) for part in new_dataset_dicts]

    return new_dataset_dicts

new_dataset_dicts_list = split_dataset_dict_equally(new_dataset_dict)

new_dataset_dicts_list=new_dataset_dicts_list[1:]

from datasets import DatasetDict, concatenate_datasets

def merge_dataset_dicts(dataset_dicts_list):
    # Initialize empty lists to hold datasets of each split
    train_datasets = []
    validation_datasets = []
    test_datasets = []
    
    # Iterate over all DatasetDicts and append the datasets of each split
    for dataset_dict in dataset_dicts_list:
        train_datasets.append(dataset_dict['train'])
        validation_datasets.append(dataset_dict['validation'])
        test_datasets.append(dataset_dict['test'])
    
    # Concatenate the datasets for each split
    merged_train = concatenate_datasets(train_datasets)
    merged_validation = concatenate_datasets(validation_datasets)
    merged_test = concatenate_datasets(test_datasets)
    
    # Create a new DatasetDict with the merged datasets
    merged_dataset_dict = DatasetDict({
        'train': merged_train,
        'validation': merged_validation,
        'test': merged_test
    })
    
    return merged_dataset_dict



# Merge the DatasetDict objects into a single one
merged_dataset_dict = merge_dataset_dicts(new_dataset_dicts_list)


from datasets import Dataset, DatasetDict

def split_dataset_dict_equally(dataset_dict, num_parts=30):
    new_dataset_dicts = [{} for _ in range(num_parts)]

    # Function to calculate split sizes and handle remainders
    def calculate_split_sizes(total_size, num_parts):
        base_size = total_size // num_parts
        remainder = total_size % num_parts
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_parts)]
        return sizes

    # Split each dataset and distribute rows evenly across new dataset dicts
    for split_name, dataset in dataset_dict.items():
        total_size = len(dataset)
        split_sizes = calculate_split_sizes(total_size, num_parts)
        offsets = [sum(split_sizes[:i]) for i in range(num_parts)]
        
        for i, offset in enumerate(offsets):
            # For the last part, take all remaining rows to ensure no row is left behind
            if i == num_parts - 1:
                part_dataset = dataset.select(range(offset, total_size))
            else:
                part_dataset = dataset.select(range(offset, offset + split_sizes[i]))
            new_dataset_dicts[i][split_name] = part_dataset

    # Wrap each dict in a DatasetDict
    new_dataset_dicts = [DatasetDict(part) for part in new_dataset_dicts]

    return new_dataset_dicts

new_dataset_dicts_list = split_dataset_dict_equally(merged_dataset_dict)






import os

import openai

# Set your OpenAI API key
openai.api_key = ""




from datasets import Dataset, DatasetDict
import pandas as pd
import openai
from openai import OpenAI

client = OpenAI()

def generate_summary_from_chunk(chunk):
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": f"Summarize the following into 20 words.Focus on core theme, key events, or major incidents, and ensure the summary is comprehensive and coherent: \"{chunk}\""
            }
        ],
        temperature=0.5
    )
    summary = summary_response.choices[0].message.content
    return summary

def add_summary_to_datasets(dataset_dict):
    def process_dataset(dataset):
        summaries = [generate_summary_from_chunk(row['chapter']) for row in dataset]
        return dataset.add_column("gpt_generated_summary", summaries)
    
    processed_datasets = {split: process_dataset(dataset_dict[split]) for split in dataset_dict.keys()}
    return DatasetDict(processed_datasets)

# Assuming `dataset_dict` is your input DatasetDict with 'train', 'validation', and 'test' datasets
# You would call the function like this:
for i in range(len(new_dataset_dicts_list)):
    
    new_dataset_dict_with_summaries = add_summary_to_datasets(new_dataset_dicts_list[i])
    # print(new_dataset_dict_with_summaries['train'][3]['gpt_generated_summary'])

    import pandas as pd

    def save_datasets_to_csv(dataset_dict):
        
        for split, dataset in dataset_dict.items():
            # Convert dataset to pandas DataFrame
            
            df = pd.DataFrame(dataset)
            # Save to CSV
            csv_path = f"{split}_dataset_with_summaries_{i}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {split} dataset to {csv_path}")

    # Assuming `new_dataset_dict_with_summaries` is your DatasetDict
    save_datasets_to_csv(new_dataset_dict_with_summaries)

