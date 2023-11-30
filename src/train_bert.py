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

from data import RibonanzaBERTDataModule
from model import RibonanzaBERT
import sys
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer

file = str(sys.argv[1])
logfolder = str(sys.argv[2])
device = ast.literal_eval(sys.argv[3])
size = str(sys.argv[4])
lr = float(sys.argv[5])
use_bpp = str(sys.argv[6]).lower() == 'true'
pos_enc = str(sys.argv[7])


size_dict = {
    "L" : [256, 8],
    "S" : [64, 4]
}

dm = RibonanzaBERTDataModule(
    file, # path to h5torch train file
    batch_size=128, # batch size for model
    n_workers=8, # num workers in dataloader
    in_memory=False,
    p=0.15,
)
dm.setup(None)

model = RibonanzaBERT(
    dim = size_dict[size][0],
    depth = size_dict[size][1],
    lr = lr,
    dropout = 0.25,
    use_bpp = use_bpp,
    pos_enc = pos_enc,
)

callbacks = [
    ModelCheckpoint(every_n_train_steps=10_000),
]
logger = TensorBoardLogger(
        logfolder,
        name="BERT",
    )

trainer = Trainer(
    accelerator="gpu",
    devices=device,
    strategy="auto",
    gradient_clip_val = 1,
    max_steps = 500_000,
    val_check_interval = 5_000,
    check_val_every_n_epoch=None,
    callbacks=callbacks,
    logger=logger,
    precision="bf16-mixed",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())