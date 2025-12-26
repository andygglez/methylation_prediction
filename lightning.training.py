
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

# Argumentos que vienen del sweep.yaml command
parser.add_argument('--npz', type=str)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--accelerator', type=str)

args = parser.parse_args()


wandb_logger = WandbLogger(project="MethPrediction")
os.environ['WANDB_API_KEY'] = '2a1829519497eaab2f05c336830a1d4b0a3a8238'
### sweep id: bmpmwvvh

run = wandb.init(
    entity="andygglez-meth",
    project="MethPrediction",
)

# Actualizar config con argumentos de línea de comandos
wandb.config.update({
    "dataset": args.npz,
    "num_workers": args.num_workers,
    "epochs": args.epochs,
    "accelerator": args.accelerator
}, allow_val_change=True)


### Prepare data
data_module = MethDataModule(npz_path=args.npz, train_split=0.8, 
                                num_workers=args.num_workers, 
                                batch_size=wandb.config.batch_size)

### Initialize model - usando parámetros del sweep
DNA_kernel_sizes = (
    wandb.config.DNA_kernel_size_1,
    wandb.config.DNA_kernel_size_2,
    wandb.config.DNA_kernel_size_3
)
DNA_strides = (
    wandb.config.DNA_stride_1,
    wandb.config.DNA_stride_2,
    wandb.config.DNA_stride_3
)

linear_layers = (
    wandb.config.linear_layer2,
    wandb.config.linear_layer3,
    wandb.config.linear_layer4
)

# Mapear nombre de optimizador a clase
optimizer_map = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}
optimizer_class = optimizer_map.get(wandb.config.optimizers, torch.optim.Adam)

model = Model(
    DNA_kernel_sizes=DNA_kernel_sizes,
    DNA_strides=DNA_strides,
    DNA_conv_channels=wandb.config.DNA_conv_channels,
    linear_layers=linear_layers,
    dropout=wandb.config.dropout,
    optimizer=optimizer_class,
    learning_rate=wandb.config.learning_rate
)

### Training the model
trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger, accelerator=args.accelerator) #, devices=-1

trainer.fit(model=model, datamodule=data_module)
trainer.test(model=model, datamodule=data_module)