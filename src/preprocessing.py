"""
Code for preprocessing data to appropriate format for the modeling.
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class MTMSequenceDataset(Dataset):
    def __init__(self, sequences, reactivity, n_out):
        self.sequences = sequences
        self.reactivity = reactivity
        self.n_out = n_out

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # convert sequence
        sequence = self.sequences[idx].ljust(self.n_out, "0")
        char_to_idx = {"0": 0, "A": 1, "C": 2, "G": 2, "U": 3} 
        sequence_numeric = [char_to_idx[char] for char in sequence]
        sequence_tensor = torch.LongTensor(sequence_numeric)
        # convert reactivity tensor
        reactivity_tensor = torch.tensor(self.reactivity.iloc[idx,:])
        # calculate mask tensor 
        mask_tensor = torch.LongTensor(np.array(~self.reactivity.iloc[idx,:].isna())*1)
        
        return sequence_tensor, reactivity_tensor, mask_tensor