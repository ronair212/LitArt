import sys
import os
# append a new directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/verma.shi/LLM/LitArt/data_module')
sys.path.append('/home/verma.shi/LLM/LitArt/models')

import argparse
import time

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

import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')

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


    #Loading the model
    model = TextSummaryModel(model=base_model,epochs=epochs,total_documents=total_documents)

    #Fitting the model
    trainer.fit(model, textmodule)

    best_model_path = checkpoint_callback.best_model_path
    print(f'Best Model Path = {best_model_path}')
