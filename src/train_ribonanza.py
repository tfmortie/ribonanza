"""
Code for training and evaluating different models.
"""
import argparse

import torch
import pandas as pd
import lightning.pytorch as pl
import ast
from torch import utils
from torch.utils.data.sampler import SubsetRandomSampler

from data import RibonanzaDataModule
from model import Transformer
import sys
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer

train_file = str(sys.argv[1])
test_file = str(sys.argv[2])
logfolder = str(sys.argv[3])
device = ast.literal_eval(sys.argv[4])
targets = str(sys.argv[5])
loss = str(sys.argv[6])
weighted = str(sys.argv[7]).lower() == 'true'
size = str(sys.argv[8])
lr = float(sys.argv[9])
use_bpp = str(sys.argv[10]).lower() == 'true'
pos_enc = str(sys.argv[11])
epochs = int(sys.argv[12])


size_dict = {
    "L" : [512, 12],
    "S" : [64, 4]
}

dm = RibonanzaDataModule(
    train_file, # path to h5torch train file
    test_file, # path to h5torch test file
    batch_size=256, # batch size for model
    n_workers=4, # num workers in dataloader
    split_number=0, # an integer 0 to 19, indicating which 1/20th (random) fraction of data to use. (for ensembling purposes later)
    train_in_memory=True,
    test_in_memory=False,
)
dm.setup(None)

model = Transformer(
    dim = size_dict[size][0],
    depth = size_dict[size][1],
    lr = lr,
    dropout = 0.25,
    loss = loss,
    weighted = weighted,
    target = targets,
    use_bpp = use_bpp,
    pos_enc = pos_enc,
)

val_ckpt = ModelCheckpoint(monitor="val_mae", mode="min")
#callbacks = [
#        val_ckpt,
#        EarlyStopping(monitor="val_mae", patience=5, mode="min"),
#    ]
logger = TensorBoardLogger(
        logfolder,
        name="trf",
    )

trainer = Trainer(
    accelerator="gpu",
    devices=device,
    strategy="auto",
    max_epochs=epochs,
    logger=logger, #callbacks=callbacks
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())