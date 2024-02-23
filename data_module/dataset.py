import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        summary = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            summary = self.target_transform(summary)
        return image, summary

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
        return len(self.chapter)

    def __getitem__(self,idx):
        chapter = "summarize:" + str(self.textprocessor.process(self.chapter[idx]))
        summary = self.textprocessor.process(self.summary[idx])

        input_encodings = self.tokenizer(chapter, max_length=self.tokenizer_chapter_max_length,padding="max_length", truncation=self.truncation)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(summary, max_length=self.tokenizer_summary_max_length,padding="max_length", truncation=self.truncation)

        return {
            "input_ids": torch.tensor(input_encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(input_encodings["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target_encodings["input_ids"], dtype=torch.long),
            "summary_mask": torch.tensor(target_encodings["attention_mask"], dtype=torch.long)
        }