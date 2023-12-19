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
from model import Transformer, RibonanzaBERT
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
batch_size = int(sys.argv[13])
ckpt_path = str(sys.argv[14])


size_dict = {
    "L" : [512, 12],
    "S" : [256, 8],
}

dm = RibonanzaDataModule(
    train_file, # path to h5torch train file
    test_file, # path to h5torch test file
    batch_size=batch_size, # batch size for model
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
    lr_decay_factor = 0.97,
    warmup_steps = 500,
)
if ckpt_path != "None":
    pretrained_model = RibonanzaBERT.load_from_checkpoint(ckpt_path)

    pretrained_trf_dict = pretrained_model.transformer.state_dict()
    pretrained_emb_dict = pretrained_model.embed.state_dict()
    pretrained_mask_dict = pretrained_model.mask_net.state_dict()


    model_trf_state_dict = model.transformer.state_dict()
    model_trf_state_dict.update(pretrained_trf_dict)
    model.transformer.load_state_dict(model_trf_state_dict)

    model_emb_state_dict = model.embed.state_dict()
    model_emb_state_dict.update(pretrained_emb_dict)
    model.embed.load_state_dict(model_emb_state_dict)

    model_mask_state_dict = model.mask_net.state_dict()
    model_mask_state_dict.update(pretrained_mask_dict)
    model.mask_net.load_state_dict(model_mask_state_dict)

val_ckpt = ModelCheckpoint(monitor="val_mae", mode="min")
callbacks = [
        val_ckpt,
        EarlyStopping(monitor="val_mae", patience=5, mode="min"),
    ]
logger = TensorBoardLogger(
        logfolder,
        name="trf",
    )

trainer = Trainer(
    accelerator="gpu",
    devices=device,
    strategy="auto",
    max_epochs=epochs,
    logger=logger,
    callbacks=callbacks,
    gradient_clip_val = 1,
    precision="bf16-mixed",
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())