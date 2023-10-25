"""
Code for preprocessing data.
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class MTMSequenceDataset(Dataset):
    def __init__(self, sequences, reactivity=None, n_out=457):
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
        sequence_tensor = torch.tensor(sequence_numeric)
        # just process sequences in case of test phase
        if self.reactivity is not None:
            # convert reactivity tensor
            reactivity_tensor = torch.tensor(np.concatenate([self.reactivity.iloc[idx,:],np.array([np.nan]*(self.n_out-len(self.reactivity.iloc[idx,:])))]))
            # calculate mask tensor 
            mask_tensor = torch.tensor(np.array(~self.reactivity.iloc[idx,:].isna()))

            return sequence_tensor, reactivity_tensor, mask_tensor
        else:
            # calculate mask tensor
            mask_tensor = (sequence_tensor!=0)
        
            return sequence_tensor, torch.tensor([]), mask_tensor