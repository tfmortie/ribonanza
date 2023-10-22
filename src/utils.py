"""
Global constants and utilities.
"""
import torch

BASE_MAPPING = {
    "0": "000",
    "A": "001",
    "C": "010",
    "G": "011",
    "U": "100"
}

# some consts
R_COLS = ["reactivity_"+f"{num:04d}" for num in range(1,207)]
RE_COLS = ["reactivity_error_"+f"{num:04d}" for num in range(1,207)]
DATA_FOLDER = "/data/thomas_gaetan_share/kaggle/ribonanza/data/"

def masked_mse_loss(input, target, reduction):
    mask = torch.isnan(target)
    out = (input[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
    
def masked_mae(input, target, reduction):
    mask = torch.isnan(target)
    # clip targets
    target_raw = torch.clip(target[~mask], 0, 1)
    # calculate MAE
    out = torch.abs(input[~mask]-target_raw)
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out

def sequence_encoder(x):
    # first pad sequence
    padding = ["0"]*((8-((len(x))%8))%8)
    padding+=x
    # now convert every base to bitstring
    base_list_bitstring = list(map(lambda x: BASE_MAPPING[x], padding))
    # convert to byte
    seq_encoded = []
    for i in range(0,len(base_list_bitstring)-7,8):
        bitstring = "".join(base_list_bitstring[i:i+8])
        # get ascii chr for every byte
        seq_encoded += [chr(int(bitstring[i:i+8],2)) for i in range(0, len(bitstring), 8)]
    seq_encoded = "".join(seq_encoded)
    
    return seq_encoded

def sequence_decoder(x):
    seq_decoded = []
    # convert ascii chr to bitstring
    bitstring = []
    for chr in x:
        bitstring.append(bin(ord(chr[0]))[2:].zfill(8))
    bitstring = "".join(bitstring)
    # now run over every base and decode
    inv_base_mapping = {BASE_MAPPING[k]:k for k in BASE_MAPPING.keys()}
    for i in range(0,len(bitstring)-2,3):
        base_bitstring = bitstring[i:i+3]
        if base_bitstring != "000":
            seq_decoded.append(inv_base_mapping[base_bitstring])
    seq_decoded = "".join(seq_decoded) 

    return seq_decoded