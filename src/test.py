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
output_file = str(sys.argv[3])
model_path = str(sys.argv[4])
device = ast.literal_eval(sys.argv[5])
targets = str(sys.argv[6])

dm = RibonanzaDataModule(
    train_file, # path to h5torch train file
    test_file, # path to h5torch test file
    batch_size=256, # batch size for model
    n_workers=4, # num workers in dataloader
    split_number=0, # an integer 0 to 19, indicating which 1/20th (random) fraction of data to use. (for ensembling purposes later)
    train_in_memory=False,
    test_in_memory=True,
)
dm.setup(None)

model = Transformer.load_from_checkpoint(model_path)

trainer = Trainer(
    accelerator="gpu",
    devices=device,
    strategy="auto",
    max_steps=100, #epochs
    callbacks=None,
    logger=None,
)

res = trainer.predict(
    model, dataloaders=dm.test_dataloader()
)

res_ids = torch.cat([res_b[0] for res_b in res]).numpy()
if targets == "both":
    res_DMS = torch.cat([res_b[1] for res_b in res]).to(torch.float32).numpy()
    res_2A3 = torch.cat([res_b[2] for res_b in res]).to(torch.float32).numpy()

    with open(output_file, "w") as f:
        for r, d, a in zip(res_ids, res_DMS, res_2A3):
            f.write("%s,%s,%s\n" % (r, d, a))

elif targets == "2A3":
    res_2A3 = torch.cat([res_b[2] for res_b in res]).to(torch.float32).numpy()

    with open(output_file, "w") as f:
        for r, a in zip(res_ids, res_2A3):
            f.write("%s,%s\n" % (r, a))

elif targets == "DMS":
    res_DMS = torch.cat([res_b[1] for res_b in res]).to(torch.float32).numpy()

    with open(output_file, "w") as f:
        for r, d in zip(res_ids, res_DMS):
            f.write("%s,%s\n" % (r, d))