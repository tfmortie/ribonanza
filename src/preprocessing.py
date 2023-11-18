"""
Code for preprocessing data.
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class MTMSequenceDataset(Dataset):
    def __init__(self, sequences, reactivity=None, reactivitye=None, n_out=457):
        self.sequences = sequences
        self.reactivity = reactivity
        self.reactivity_error = reactivitye
        self.n_out = n_out

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # convert sequence
        sequence = self.sequences.iloc[idx].ljust(self.n_out, "0")
        char_to_idx = {"0": 0, "A": 1, "C": 2, "G": 2, "U": 3} 
        sequence_numeric = [char_to_idx[char] for char in sequence]
        sequence_tensor = torch.tensor(sequence_numeric)
        # just process sequences in case of test phase
        if self.reactivity is not None:
            # convert reactivity tensors
            reactivity_tensor = torch.tensor(np.concatenate([self.reactivity.iloc[idx,:],np.array([np.nan]*(self.n_out-len(self.reactivity.iloc[idx,:])))]))
            reactivity_error_tensor = torch.tensor(np.concatenate([self.reactivity_error.iloc[idx,:],np.array([np.nan]*(self.n_out-len(self.reactivity_error.iloc[idx,:])))]))
            # calculate mask tensor 
            mask_tensor = torch.tensor(np.array(~self.reactivity.iloc[idx,:].isna()))

            return sequence_tensor, reactivity_tensor, reactivity_error_tensor, mask_tensor
        else:
            # calculate mask tensor
            mask_tensor = (sequence_tensor!=0)
        
            return sequence_tensor, torch.tensor([]), torch.tensor([]), mask_tensor

def random_split(data, val_ratio=0.2, seed=2023):
    train_idx, val_idx = [], []
    # extract subset of data that contains non-unique sequences
    data_nu_idx = list(np.arange(len(data))[data['sequence_id'].duplicated(keep=False)])
    data_nu = data.loc[data['sequence_id'].duplicated(keep=False),:]
    # extract subset of data that contains unique sequences
    data_u_idx = list(np.arange(len(data))[~data['sequence_id'].duplicated(keep=False)])
    data_u = data.loc[~data['sequence_id'].duplicated(keep=False),:]
    # update index 
    data_nu.index = data_nu_idx 
    data_u.index = data_u_idx 
    # split data_u in two parts based on val_ratio
    shuffled_data_u = data_u.sample(frac=1, random_state=seed) 
    split_index_data_u = int(len(shuffled_data_u) * val_ratio)
    # add idx to lists
    val_idx.extend(list(shuffled_data_u.iloc[:split_index_data_u].index))
    train_idx.extend(list(shuffled_data_u.iloc[split_index_data_u:].index))
    # split data_nu in two deterministic parts based on ordered signal_to_noise
    # sort first 
    data_nu_sorted = data_nu.sort_values(by=['sequence_id', 'signal_to_noise'], ascending=[True, True])
    for idx in data_nu_sorted.sequence_id.unique():
        idx_subset = data_nu_sorted.loc[data_nu_sorted.sequence_id==idx,:]
        # just conside a 50/50 deterministic split
        split_index_subset = int(len(idx_subset)*0.5)
        train_idx.extend(list(idx_subset.iloc[:split_index_subset].index))
        val_idx.extend(list(idx_subset.iloc[split_index_subset:].index))

    return train_idx, val_idx