
import sys
import os
# append a new directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/patil.adwa/LLM/LitArt/')
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

import glob
import os
import shutil
from datasets import  load_dataset

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

import pandas as pd
import lightning as L
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

from data_module.dataset import TextSummaryDataset
from data_module.data_preprocessor import TextPreprocessing

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataModule(L.LightningDataModule):
    def __init__(self,
                 train_path,
                 test_path,
                 val_path,
                 textprocessor,
                 tokenizer,
                 causal=False,
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
        if causal == True:
            self.tokenizer.pad_token = tokenizer.eos_token
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

    def total_documents(self):
        
        total_documents = self.train_df.shape[0] + self.test_df.shape[0] + self.val_df.shape[0]

        return total_documents


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
    def __init__(self,
                 tokenizer,
                 og_dataset:pd.DataFrame,
                 batch_size:int,
                 dataloader_num_workers:int=-1):
        super().__init__()
        self.og_dataset = og_dataset
        self.train_dataset = None
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.tokenizer = tokenizer
    
    def tokenize_captions(self,
                          examples, 
                          is_train=True):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def prepare_data(self,
                    destination_path:str="../data/genre_fantasy/"):

        img_paths = self.og_dataset.imageLocation.tolist()
        if os.path.exists(destination_path) and os.path.isdir(destination_path):
            pass
        else: 
            os.makedirs(destination_path, exist_ok=True)

            for img_path in img_paths[4:]:
                try:
                    filename = os.path.basename(img_path)
                    extension = os.path.splitext(filename)[1]

                    new_filename = f"{filename[:-len(extension)]}{extension}"

                    new_destination_path = os.path.join(destination_path, new_filename)

                    shutil.copy2(img_path, new_destination_path)
                except Exception as e:
                    print(f"Error transferring {img_path}: {e}")
            
            metadata = pd.DataFrame({'file_name':self.og_dataset.imageLocation.tolist(),'text':self.og_dataset.Synopsis.tolist()})
            metadata.dropna(subset=['file_name'],inplace=True)
            metadata.file_name = metadata.file_name.apply(lambda x: os.path.basename(x))
            metadata.to_csv(destination_path+"metadata.csv",index=False)
        print("Dataset is ready to Load!!")
    
    def preprocess_train(self,examples):
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.tokenize_captions(examples)
        return examples
    
    def collate_fn(self,examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def setup(self,
              accelerator,
              train_data_dir:str,
              resolution:int,
              center_crop:bool=False,
              max_train_samples:int=100,
              cache_dir:str='./',
              seed:int=101,
              ):
        ## Image Data
        data_files = {}
        if train_data_dir is not None:
            data_files["train"] = os.path.join(train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cache_dir,
        )
        print("-------------")
        print(f"Dataset:{dataset}")
        print("-------------")
        column_names = dataset['train'].column_names

        self.image_column = column_names[0]

        self.caption_column = column_names[1]
        
        ## Image transformations
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )

        with accelerator.main_process_first():
            if max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))

            # Set the training transforms
            self.train_dataset = dataset["train"].with_transform(self.preprocess_train)
        
        


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          num_workers=self.dataloader_num_workers)

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

