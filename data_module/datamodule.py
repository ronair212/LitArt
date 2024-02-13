import dataset
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import lightning as L
from data_preprocessor import TextPreprocessing
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"



class TextSummaryDataset(Dataset):
    def __init__(self,
                 df,
                 textprocessor,
                 tokenizer,
                 tokenizer_chapter_max_length=1024,
                 tokenizer_summary_max_length=64,
                 truncation=True,
                 ):

        self.df = df
        self.textprocessor = textprocessor
        self.chapter = df["chapter"]
        self.summary = df["summary_text"]
        self.tokenizer = tokenizer
        self.tokenizer_chapter_max_length = tokenizer_chapter_max_length
        self.tokenizer_summary_max_length = tokenizer_summary_max_length
        self.truncation = truncation


        def __len__(self):
            return len(df)

        def __getitem__(self,idx):
            chapter = "summarize:" + str(textprocessor.process(self.chapter[idx]))
            summary = textprocessor.process(self.summary[idx])

            input_encodings = tokenizer(chapter, max_length=self.tokenizer_chapter_max_length, truncation=self.truncation)

            with tokenizer.as_target_tokenizer():
                target_encodings = tokenizer(summary, max_length=self.tokenizer_summary_max_length, truncation=self.truncation)

            return {
                "input_ids": torch.tensor(input_encodings["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(input_encodings["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(target_encodings["input_ids"], dtype=torch.long),
                "summary_mask": torch.tensor(target_encodings["attention_mask"], dtype=torch.long)
            }


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

        # Reading the train file
        try:
            self.train_df = pd.read_csv(train_path)
        except Exception as e:
            print(f"Exception raised while reading training file at path : {train_path} \n Exception : {e}")

        # Reading the test file
        try:
            self.test_df = pd.read_csv(test_path)
        except Exception as e:
            print(f"Exception raised while reading test file at path : {test_path} \n Exception : {e}")

        # Reading the validation file
        try:
            self.val_df = pd.read_csv(val_path)
        except Exception as e:
            print(f"Exception raised while reading validation file at path : {val_path} \n Exception : {e}")

        # Textprocessor setup
        self.textprocessor = textprocessor

        # Tokenizer setup
        self.tokenizer = tokenizer
        self.tokenizer_chapter_max_length = tokenizer_chapter_max_length
        self.tokenizer_summary_max_length = tokenizer_summary_max_length
        self.truncation = truncation

        # Batch size setup
        self.batch_size = batch_size


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
            num_workers=4)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

class ImageDataModule(L.LightningModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

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



# Training files setup
train_path = "../Datasets/Training_data.csv"
test_path = "../Datasets/Testing_data.csv"
val_path = "../Datasets/Validation_data.csv"

# Text Preprocessor setup
textpreprocessor = TextPreprocessing()

# Tokenizer Setup
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

textmodule = TextDataModule(train_path=train_path,
                                     val_path=val_path,
                                     test_path=test_path,
                                     textprocessor=textpreprocessor,
                                     tokenizer=tokenizer,
                                     tokenizer_chapter_max_length=1024,
                                     tokenizer_summary_max_length=64,
                                     truncation=True)
textmodule.setup()

