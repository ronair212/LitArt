import sys
import os
# append a new directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/verma.shi/LLM/LitArt/data_module')
sys.path.append('/home/verma.shi/LLM/LitArt/models')

import argparse
import time
import json
from datetime import date

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping

from data_module.data_preprocessor import TextPreprocessing
from data_module.dataset import TextSummaryDataset
from data_module.datamodule import TextDataModule

from models.summarizer import TextSummaryModel

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from lightning.pytorch.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

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
                    type=int,help='Batchsize')
    parser.add_argument('--chapterlength',
                    type=int,help='Chapter Length')
    parser.add_argument('--summarylength',
                    type=int,help='Summary Length')
    parser.add_argument('--num_epochs',
                    type=int,help='Number of epochs')

    #Logging details
    parser.add_argument('--log_path',
                type=str,help='Path to save logs')

    #Model Cache dir
    parser.add_argument('--cache_dir',
                type=str,help="Cache directory location")

    args = parser.parse_args()

    #Loading the data
    train_path = args.trainpath
    test_path = args.testpath
    val_path = args.valpath

    #Loading the model and tokenizer
    base_model_name = args.model
    tokenizer_name = args.tokenizer
    cache_dir = args.cache_dir
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,cache_dir=cache_dir)

    #Fine Tuning Parameters
    tokenizer_chapter_max_length = args.chapterlength
    tokenizer_summary_max_length = args.summarylength

    #Training Parameters
    batch_size = args.batchsize
    epochs = args.num_epochs
    log_path = args.log_path

    #Initializing the dataloaders
    textpreprocessor = TextPreprocessing()
    textdatamodule = TextDataModule(train_path=train_path,
                                     val_path=val_path,
                                     test_path=test_path,
                                     textprocessor=textpreprocessor,
                                     tokenizer=tokenizer,
                                     tokenizer_chapter_max_length=tokenizer_chapter_max_length,
                                     tokenizer_summary_max_length=tokenizer_summary_max_length,
                                     batch_size=batch_size,
                                     truncation=True)

    textdatamodule.prepare_data()
    textdatamodule.setup()
    total_documents = textdatamodule.total_documents()

    today = date.today()

    #Simplified Log 
    log_path = log_path+base_model_name.replace("/","-")+"-" +str(today)+"-"+time.strftime("%H:%M:%S", time.localtime())
    logger = TensorBoardLogger(log_path, name="my_model")
    
    run_config = {
        "train_path":args.trainpath,
        "test_path":args.testpath,
        "val_path":args.valpath,
        "base_model_name":args.model,
        "tokenizer_name":args.tokenizer,
        "cache_dir":args.cache_dir,
        "batch_size":args.batchsize,
        "tokenizer_chapter_max_length":args.chapterlength,
        "tokenizer_summary_max_length":args.summarylength,
        "batch_size":args.batchsize,
        "epochs":args.num_epochs,
        "log_path":args.log_path,
    }



    # File path
    filname = 'run_config.json'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Write the dictionary to a file in JSON format
    with open(log_path+"/"+filname, 'w+') as file:
        json.dump(run_config, file)


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
        default_root_dir = log_path,
        logger=logger
    )


    #Loading the model
    model = TextSummaryModel(model=base_model,epochs=epochs,total_documents=total_documents)

    #Fitting the model
    trainer.fit(model, textdatamodule)

    best_model_path = checkpoint_callback.best_model_path
    print(f'Best Model Path = {best_model_path}')
