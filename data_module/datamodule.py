import dataset
from torch.utils.data import DataLoader
import pandas as pd
import lightning as L

class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        self.train = pd.read_csv('./train.csv')
        self.val = pd.read_csv('./val.csv')
        self.test = pd.read_csv('./test.csv')

    def setup(self, stage: str):
        ## Text Data
        self.traindataset = dataset.TextSummaryDataset(self.train['chapters'],
                                                self.train['summary'])
        self.valdataset = dataset.TextSummaryDataset(self.val['chapters'],
                                                self.val['summary'])
        self.testdataset = dataset.TextSummaryDataset(self.test['chapters'],[])

    def train_dataloader(self):
        return DataLoader(self.traindataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valdataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testdataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...