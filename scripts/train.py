import sys
# append a new directory to sys.path
sys.path.append('/home/verma.shi/LLM/LitArt/')

import argparse
import os
import time

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping

# from data_module.data_preprocessor import TextPreprocessing
from data_module.dataset import TextSummaryDataset
from data_module.datamodule import TextDataModule
from models.summarizer import TextSummaryModel


import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup, AdamW

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    #LLM Model Details
    parser.add_argument('--model' ,
                        type=str, help='Model to fine tune')
    parser.add_argument('--tokenizer',
                        type=str,help='Tokenizer to use')

    #Data folder location
    parser.add_argument('--trainpath',
                    type=str,help='Location for training file')
    parser.add_argument('--testpath',
                    type=str,help='Location for testing file')
    parser.add_argument('--valpath',
                    type=str,help='Location for validation file')

    #Fine Tunining Parameters
    parser.add_argument('--batchsize',
                    type=str,help='Batchsize')
    parser.add_argument('--chapterlength',
                    type=str,help='Chapter Length')
    parser.add_argument('--summarylength',
                    type=str,help='Summary Length')
    parser.add_argument('--num_epochs',
                    type=str,help='Number of epochs')

    #Logging details
    parser.add_argument('--log_path',
                type=str,help='Path to save logs')

    args = parser.parse_args()


    #Loading the data
    train_path = args.trainpath
    test_path = args.testpath
    val_path = args.valpath

    #Initializing the dataloaders
    textpreprocessor = TextPreprocessing()
    textmodule = TextDataModule(train_path=train_path,
                                     val_path=val_path,
                                     test_path=test_path,
                                     textprocessor=textpreprocessor,
                                     tokenizer=tokenizer,
                                     tokenizer_chapter_max_length=1024,
                                     tokenizer_summary_max_length=64,
                                     truncation=True)

    textmodule.prepare_data()
    textmodule.setup()
    total_documents = textmodule.total_documents()

    #Model Parameters
    base_model_name = args.model
    tokenizer = args.tokenizer
    batch_size = args.batchsize
    chapter_length = args.chapterlength
    summary_length = args.summarylength
    epochs = args.num_epochs
    log_path = args.log_path

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_top_k=1,
    )

    trainer = L.Trainer(
        callbacks=[
                    checkpoint_callback,
                ],
        max_epochs = epochs,
        accelerator="gpu",
        devices=1,
        default_root_dir = log_path
    )

    tokenizer = AutoModelForSeq2SeqLM.from_pretrained(model)
    base_model = AutoTokenizer.from_pretrained("t5-small")





    print(model_name)