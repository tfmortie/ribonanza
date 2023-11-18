"""
Code for testing different models.
"""
import argparse
import torch

import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DATA_FOLDER
from model import MTM, MTMModel, Conv1DBaseline
from preprocessing import MTMSequenceDataset

def test_mtm(args):
    # read in data
    print("Reading in data")
    data = pd.read_csv(DATA_FOLDER+args.data)
    print("Done!")
    # init our mtm model
    mtm_arch = MTM(args.embeddingsize, args.mtmhidden1, args.mtmhidden2, args.seqlen)
    # create dataset and dataloaders
    mtmseq_data = MTMSequenceDataset(data.sequence, None, None, args.seqlen)
    test_loader = DataLoader(mtmseq_data, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    # init the our mtm model
    model = MTMModel.load_from_checkpoint(args.modelpath, model=mtm_arch)
    # put in evaluation mode
    print("Start calculating predictions...")
    model.eval()
    preds = [] 
    for i, (batch_seq, _, batch_mask) in tqdm(enumerate(test_loader)):
        batch_seq = batch_seq.long()
    
        out = model.model(batch_seq)
        preds.extend(list(out[batch_mask].detach().numpy()))
    print(len(preds))
    out_df = pd.DataFrame({"Pred": preds})
    out_df.to_csv("./"+args.out+".csv")
    print("Done!")

def test_conv(args):
    # read in data
    print("Reading in data")
    data = pd.read_csv(DATA_FOLDER+args.data)
    print("Done!")
    # init our mtm model
    mtm_arch = Conv1DBaseline(args.embeddingsize, args.n_layers)
    # create dataset and dataloaders
    mtmseq_data = MTMSequenceDataset(data.sequence, None, None, args.seqlen)
    test_loader = DataLoader(mtmseq_data, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers)
    # init the our mtm model
    model = MTMModel.load_from_checkpoint(args.modelpath, model=mtm_arch)
    # put in evaluation mode
    print("Start calculating predictions...")
    model.eval().to("cuda:%s" % args.device)
    preds = []
    with torch.no_grad():
        for i, (batch_seq, _, batch_mask) in tqdm(enumerate(test_loader)):
            batch_seq = batch_seq.long().to("cuda:%s" % args.device)

            out = model.model(batch_seq)
            preds.extend(list(out[batch_mask].detach().cpu().numpy()))
    print(len(preds))
    out_df = pd.DataFrame({"Pred": preds})
    out_df.to_csv("./"+args.out+".csv")
    print("Done!")

TEST_MODEL = {
    "mtm": test_mtm,
    "conv" : test_conv,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test code')
    """ model and data params """
    parser.add_argument("-d", "--data", type=str, required=True, help="filename of test dataset")
    parser.add_argument("-m", "--model", type=str, default="mtm", choices=["mtm", "conv"], help="model of interest")
    parser.add_argument("-mp", "--modelpath", type=str, required=True, help="path to saved model")
    parser.add_argument("-embs", "--embeddingsize", type=int, default=3, help="size of embeddings")
    parser.add_argument("-mtmh1", "--mtmhidden1", type=int, default=64, help="size of first hidden layer for MTM model")
    parser.add_argument("-mtmh2", "--mtmhidden2", type=int, default=128, help="size of second hidden layer for MTM model")
    parser.add_argument("-seqlen", "--seqlen", type=int, default=206, help="length of sequences to consider")
    parser.add_argument("-nl", "--n_layers", type=int, default=10, help="number of layers in conv baseline")
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("-nw", "--numworkers", type=int, default=3, help="number of workers")
    parser.add_argument("-dev", "--device", type=int, default=1, help="gpu device to use")
    """ logging """
    parser.add_argument("-out", "--out", type=str, default="submission",help="filename for predictions")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose param")
    """ other """
    parser.add_argument("-s", "--seed", type=int, default=2023, help="seed for rng")
    args = parser.parse_args()
    TEST_MODEL[args.model](args)