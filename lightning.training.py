
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
import os
import argparse

from modules.data_modules import MethDataModule
from modules.model import Model


parser = argparse.ArgumentParser()

parser.add_argument('--npz', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--accelerator', type=str)


args = parser.parse_args()


wandb_logger = WandbLogger(project="MethPrediction")
# os.environ['WANDB_API_KEY'] = '2a1829519497eaab2f05c336830a1d4b0a3a8238'

# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="andygglez-meth",
#     # Set the wandb project where this run will be logged.
#     project="MethPrediction",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": 1e-3,
#         "architecture": "CNN+ATT",
#         "dataset": "chr19.npz",
#         "epochs": 5,
#     },
# )


### Prepare data
data_module = MethDataModule(npz_path=args.npz, train_split=0.8, num_workers=args.num_workers, batch_size=args.batch_size)

### Initialize model
model = Model(DNA_kernel_sizes=(10,10,5), DNA_strides=(2,3,3), DNA_conv_channels = 2)

### Training the model
trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger, accelerator=args.accelerator) #, devices=-1

trainer.fit(model=model, datamodule=data_module)
trainer.test(model=model, datamodule=data_module)