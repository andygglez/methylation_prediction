
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
import os


from modules.data_modules import MethDataModule

# Enable Tensor Cores for H100
torch.set_float32_matmul_precision('high')

wandb_logger = WandbLogger(project="MethPrediction")
os.environ['WANDB_API_KEY'] = '2a1829519497eaab2f05c336830a1d4b0a3a8238'


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


#### Dataset Class
## Notice that the __init__ method contains an argument `apply_log10`, if you set it to True
## you will apply a log10 to the raw counts. We can experiment with this

### Prepare data
data_module = MethDataModule(npz_path='chr19.npz', train_split=0.8, batch_size=32)

### Initialize model
model = Model(DNA_kernel_sizes=(10,10,5), DNA_strides=(2,3,3), DNA_conv_channels = 2)

### Training the model
trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, accelerator="gpu", devices=-1)
model = torch.compile(model)

trainer.fit(model=model, train_dataloaders=data_module)
trainer.test(model=model, dataloaders=data_module)