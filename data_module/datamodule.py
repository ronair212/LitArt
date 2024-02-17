import glob
import pandas as pd

import dataset
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import lightning as L
from data_preprocessor import TextPreprocessing
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataModule(L.LightningDataModule):
    def __init__(self,
                 train_path,
                 test_path,
                 val_path,
                 textprocessor,
                 tokenizer,
                 tokenizer_chapter_max_length=1024,
                 tokenizer_summary_max_length=64,
                 truncation = True,
                 batch_size: int = 32):


        super().__init__()

        # Initializing Paths
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        # Initializing Dataframes
        self.train_df = None
        self.test_df = None
        self.val_df = None

        # Textprocessor setup
        self.textprocessor = textprocessor

        # Tokenizer setup
        self.tokenizer = tokenizer
        self.tokenizer_chapter_max_length = tokenizer_chapter_max_length
        self.tokenizer_summary_max_length = tokenizer_summary_max_length
        self.truncation = truncation

        # Batch size setup
        self.batch_size = batch_size

    def prepare_data(self):
         # Reading the train file
        try:
            self.train_df = pd.read_csv(self.train_path)
        except Exception as e:
            print(f"Exception raised while reading training file at path : {self.train_path} \n Exception : {e}")

        # Reading the test file
        try:
            self.test_df = pd.read_csv(self.test_path)
        except Exception as e:
            print(f"Exception raised while reading test file at path : {self.test_path} \n Exception : {e}")

        # Reading the validation file
        try:
            self.val_df = pd.read_csv(self.val_path)
        except Exception as e:
            print(f"Exception raised while reading validation file at path : {self.val_path} \n Exception : {e}")


    def setup(self, stage= None):
        self.train_dataset = TextSummaryDataset(
            df=self.train_df,
            textprocessor=self.textprocessor,
            tokenizer=self.tokenizer,
            tokenizer_chapter_max_length=self.tokenizer_chapter_max_length,
            tokenizer_summary_max_length=self.tokenizer_summary_max_length,
            truncation=self.truncation)

        self.val_dataset = TextSummaryDataset(
            df=self.val_df,
            textprocessor=self.textprocessor,
            tokenizer=self.tokenizer,
            tokenizer_chapter_max_length=self.tokenizer_chapter_max_length,
            tokenizer_summary_max_length=self.tokenizer_summary_max_length,
            truncation=self.truncation)

        self.test_dataset = TextSummaryDataset(
            df=self.test_df,
            textprocessor=self.textprocessor,
            tokenizer=self.tokenizer,
            tokenizer_chapter_max_length=self.tokenizer_chapter_max_length,
            tokenizer_summary_max_length=self.tokenizer_summary_max_length,
            truncation=self.truncation)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0
        )

class ImageDataModule(L.LightningModule):
    def __init__(self, data_dir: str = "path/to/dir",
                 batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        image_paths = glob.glob(data_dir+'bookcovers/*.jpg')
        text = pd


    def setup(self, stage: str):
        ## Image Data
        pass

    def train_dataloader(self):
        return DataLoader(self.traindataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valdataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testdataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


# # Training files setup
# train_path = "../Datasets/Training_data.csv"
# test_path = "../Datasets/Testing_data.csv"
# val_path = "../Datasets/Validation_data.csv"

# # Text Preprocessor setup
# textpreprocessor = TextPreprocessing()

# # Tokenizer Setup
# model_ckpt = "google/pegasus-cnn_dailymail"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# # model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# textmodule = TextDataModule(train_path=train_path,
#                                      val_path=val_path,
#                                      test_path=test_path,
#                                      textprocessor=textpreprocessor,
#                                      tokenizer=tokenizer,
#                                      tokenizer_chapter_max_length=1024,
#                                      tokenizer_summary_max_length=64,
#                                      truncation=True)
# textmodule.setup()

