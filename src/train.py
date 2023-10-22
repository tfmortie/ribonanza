"""
Code for training and evaluating different models.
"""
import argparse
import torch

import pandas as pd
import lightning.pytorch as pl

from torch import utils

from utils import DATA_FOLDER, R_COLS
from model import MTM, MTMModel
from preprocessing import MTMSequenceDataset

def train_mtm(args):
    # read in data
    print("Reading in data")
    data = pd.read_csv(DATA_FOLDER+args.data)
    print("Done!")
    # init our mtm model
    mtm_arch = MTM(3, args.mtmhidden1, args.mtmhidden2, args.seqlen)
    # init the our mtm model
    mtm_model = MTMModel(mtm_arch)
    # create dataset and dataloaders
    mtmseq_data = MTMSequenceDataset(data.sequence, data.loc[:,R_COLS], 206)
    # create generator
    mtmseq_train_set, mtmseq_val_set = torch.utils.data.random_split(mtmseq_data, [0.8, 0.2], generator = torch.Generator().manual_seed(args.seed))
    # and split in training and validation
    train_loader = utils.data.DataLoader(mtmseq_train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers)
    val_loader = utils.data.DataLoader(mtmseq_val_set, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    # train
    trainer = pl.Trainer(limit_train_batches=args.maxbratio, max_epochs=args.nepochs, devices=args.numdevices)
    trainer.fit(model=mtm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

TRAIN_MODEL = {
    "mtm": train_mtm,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    """ model and data params """
    parser.add_argument("-d", "--data", type=str, required=True, help="filename of dataset")
    parser.add_argument("-m", "--model", type=str, default="mtm", choices=["mtm"], help="model of interest")
    parser.add_argument("-mtmh1", "--mtmhidden1", type=int, default=64, help="size of first hidden layer for MTM model")
    parser.add_argument("-mtmh2", "--mtmhidden2", type=int, default=128, help="size of second hidden layer for MTM model")
    parser.add_argument("-seqlen", "--seqlen", type=int, default=206, help="length of sequences to consider")
    """ training params """
    parser.add_argument("-maxbr", "--maxbratio", type=int, default=1.0, help="ratio of training batches to consider")
    parser.add_argument("-nepochs", "--nepochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("-lr", "--learnrate", type=float, default=0.01, help="learning rate")
    parser.add_argument("-nw", "--numworkers", type=int, default=3, help="number of workers")
    parser.add_argument("-nd", "--numdevices", type=int, default=1, help="number of devices") # currently set to one due to memory leaks (TODO solve)
    """ logging """
    parser.add_argument("-out", "--out", type=str, default="my_model.pt", help="filename of stored model")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose param")
    """ other """
    parser.add_argument("-s", "--seed", type=int, default=2023, help="seed for rng")
    args = parser.parse_args()
    TRAIN_MODEL[args.model](args)