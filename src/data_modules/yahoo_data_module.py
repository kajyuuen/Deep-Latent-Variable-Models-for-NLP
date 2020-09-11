from typing import Optional

import pytorch_lightning as pl
import torchtext.data as data

from src.datasets.yahoo_dataset import YahooDataset

class YahooDataModule(pl.LightningDataModule):
    def __init__(self,
                 path: str,
                 batch_size: int = 5000):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self):
        self.text = data.Field()
        self.dataset = YahooDataset(self.text, path = self.path)
        self.text.build_vocab(self.dataset, max_size=5000)
        self.vocab_size = len(self.text.vocab.itos)

    def train_dataloader(self):
        train_iter = next(iter(data.BucketIterator(
                dataset=self.dataset, batch_size=self.batch_size,
            sort_key=lambda x: len(x.text))))
        return train_iter
