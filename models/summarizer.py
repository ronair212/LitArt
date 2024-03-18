import sys
# append a new directory to sys.path
sys.path.append('/home/verma.shi/LLM/LitArt/')
import dataset
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
from data_module.data_preprocessor import TextPreprocessing
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
import re

class TextSummaryModel(L.LightningModule):
    def __init__(self,model,
                     total_documents = 5000,
                     epochs=2):
        super(TextSummaryModel,self).__init__()
        self.model = model
        self.epochs = int(epochs)
        self.total_documents = int(total_documents)


    def set_model(self,model):
        self.model = model

    def forward(self, 
                input_ids, 
                attention_mask, 
                labels = None, 
                decoder_attention_mask = None):
        
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_attention_mask=decoder_attention_mask)

        return outputs.loss, outputs.logits

    def training_step(self,batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["summary_mask"]

        loss , output = self(input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels,
                            decoder_attention_mask = decoder_attention_mask)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self , batch , batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["summary_mask"]

        loss , output = self(input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels,
                            decoder_attention_mask = decoder_attention_mask)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["summary_mask"]
        loss, output = self(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            labels = labels,
                            decoder_attention_mask = decoder_attention_mask)
        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.001,weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=500,
                num_training_steps=self.epochs*self.total_documents)
        return {'optimizer': optimizer,'lr_scheduler': scheduler}

