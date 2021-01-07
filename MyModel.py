import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MyModel(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        with open("saved.config.yaml", "w", encoding="utf-8") as fp:
            OmegaConf.save(hparams, fp, resolve=True)
        self.encoder = torch.nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        return 0.0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(
            MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
        )
