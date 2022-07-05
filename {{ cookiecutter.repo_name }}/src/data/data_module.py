"""
Example of data module to train and validate on FashionMNIST
For referene, the ID to category correspondence is :
id_to_label = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
"""
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.data.dataset import get_dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def prepare_data(self):
        self.train_dataset = get_dataset(train=True)
        self.valid_dataset = get_dataset(train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.params['train_dataloader'])

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.params['val_dataloader'])
