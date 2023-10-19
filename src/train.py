"""
Code for training and evaluating different models.
"""
import argparse

def train_mtm(args):
    print("Train a mtm model!")
    print(args)

TRAIN_MODEL = {
    "mtm": train_mtm,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    """ model and data params """
    parser.add_argument("-t", "--target", type=str, default="DMS", choices=["DMS", "2A3"], help="target of interest")
    parser.add_argument("-m", "--model", type=str, default="mtm", choices=["mtm"], help="model of interest")
    parser.add_argument("-maxn", "--maxn", type=int, default=-1, help="max number of training instances")
    """ training params """
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("-lr", "--learnrate", type=float, default=0.01, help="learning rate")
    """ logging """
    parser.add_argument("-out", "--out", type=str, default="my_model.pt", help="filename of stored model")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose param")
    args = parser.parse_args()
    TRAIN_MODEL[args.model](args)