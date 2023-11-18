"""
Code for training and evaluating different models.
"""
import argparse
import torch

import pandas as pd
import lightning.pytorch as pl

from torch import utils
from torch.utils.data.sampler import SubsetRandomSampler

from utils import DATA_FOLDER, R_COLS, RE_COLS
from model import MTM, Conv1DBaseline, MTMModel
from preprocessing import MTMSequenceDataset, random_split


def train_mtm(args):
    # read in data
    print("Reading in data")
    df_list = []
    for f in args.data:
        df_list.append(pd.read_csv(DATA_FOLDER+f))
    data = pd.concat(df_list)
    print("Done!")
    # init our mtm model
    mtm_arch = MTM(args.embeddingsize, args.mtmhidden1, args.mtmhidden2, args.seqlen)
    # init the our mtm model
    mtm_model = MTMModel(mtm_arch)
    # create dataset and dataloaders
    mtmseq_data = MTMSequenceDataset(data.sequence, data.loc[:,R_COLS], data.loc[:,RE_COLS], args.seqlen)
    # get splits
    train_idx, val_idx = random_split(data, args.valratio, seed=args.seed) 
    mtmseq_train_set, mtmseq_val_set = torch.utils.data.Subset(mtmseq_data, train_idx), torch.utils.data.Subset(mtmseq_data, val_idx)
    # and split in training and validation
    train_loader = utils.data.DataLoader(mtmseq_train_set, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    val_loader = utils.data.DataLoader(mtmseq_val_set, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    # train
    print("Start training model...")
    trainer = pl.Trainer(limit_train_batches=args.maxbratio, max_epochs=args.nepochs, devices=args.numdevices)
    trainer.fit(model=mtm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # save model
    trainer.save_checkpoint("./{}.ckpt".format(args.out))
    print("Done!")

def train_conv(args):
    # read in data
    print("Reading in data")
    df_list = []
    for f in args.data:
        df_list.append(pd.read_csv(DATA_FOLDER+f))
    data = pd.concat(df_list)
    print("Done!")
    # init our mtm model
    mtm_arch = Conv1DBaseline(args.embeddingsize, args.n_layers)
    # init the our mtm model
    mtm_model = MTMModel(mtm_arch, lr = args.learningrate)
    # create dataset and dataloaders
    mtmseq_data = MTMSequenceDataset(data.sequence, data.loc[:,R_COLS], data.loc[:,RE_COLS], args.seqlen)
    # get splits
    train_idx, val_idx = random_split(data, args.valratio, seed=args.seed)
    mtmseq_train_set, mtmseq_val_set = torch.utils.data.Subset(mtmseq_data, train_idx), torch.utils.data.Subset(mtmseq_data, val_idx)
    # and create dataloaders
    train_loader = utils.data.DataLoader(mtmseq_train_set, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    val_loader = utils.data.DataLoader(mtmseq_val_set, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    # train
    print("Start training model...")
    trainer = pl.Trainer(limit_train_batches=args.maxbratio, max_epochs=args.nepochs, devices=[args.device]) # I changed the devices arg so that it now specifies which gpu
    trainer.fit(model=mtm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # save model
    trainer.save_checkpoint("./{}.ckpt".format(args.out))
    print("Done!")

TRAIN_MODEL = {
    "mtm": train_mtm,
    "conv" : train_conv,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training code')
    """ model and data params """
    parser.add_argument("-d", "--data", type=str, required=True, nargs='+', help="filename(s) of dataset")
    parser.add_argument("-m", "--model", type=str, default="mtm", choices=["mtm", "conv"], help="model of interest")
    parser.add_argument("-embs", "--embeddingsize", type=int, default=3, help="size of embeddings")
    parser.add_argument("-mtmh1", "--mtmhidden1", type=int, default=64, help="size of first hidden layer for MTM model")
    parser.add_argument("-mtmh2", "--mtmhidden2", type=int, default=128, help="size of second hidden layer for MTM model")
    parser.add_argument("-sl", "--seqlen", type=int, default=457, help="length of sequences to consider")
    parser.add_argument("-nl", "--n_layers", type=int, default=10, help="number of layers in conv baseline")
    """ training params """
    parser.add_argument("-vr", "--valratio", type=float, default=0.2, help="ratio of training data to consider for validation")
    parser.add_argument("-mb", "--maxbratio", type=int, default=1.0, help="ratio of training batches to consider")
    parser.add_argument("-ne", "--nepochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("-nw", "--numworkers", type=int, default=3, help="number of workers")
    parser.add_argument("-lr", "--learningrate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("-dev", "--device", type=int, default=1, help="which gpu device to use")
    """ logging """
    parser.add_argument("-out", "--out", type=str, default="my_model", help="filename for model")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose param")
    """ other """
    parser.add_argument("-s", "--seed", type=int, default=2023, help="seed for rng")
    args = parser.parse_args()
    TRAIN_MODEL[args.model](args)