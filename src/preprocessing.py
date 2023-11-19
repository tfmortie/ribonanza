import pandas as pd
import os
import numpy as np
import h5torch
import sys
"""
Script to generate h5torch file for train dataset.
"""

train_data_file = sys.argv[1] # e.g. "./data/train_data_new.csv"
bpp_folder = sys.argv[2] # e.g. "./data/Ribonanza_bpp_files/extra_data/"
train_data_output_file = sys.argv[3] # e.g. "./data/train.h5t"

data = pd.read_csv(train_data_file)

data_2A3 = data[data["experiment_type"] == "2A3_MaP"]
data_DMS = data[data["experiment_type"] == "DMS_MaP"]

bpp_files = {}
for (dirpath, dirnames, filenames) in os.walk(bpp_folder):
    for file in filenames:
        #print(dirpath, dirnames, file)
        bpp_files[file.rstrip(".txt")] = os.path.join(dirpath, file)

nucleotide_mapper = {"0": 0, "A": 1, "C": 2, "G": 3, "U": 4}


# generate dataset objects:
ids = []
sequences = []
s_to_n_2A3 = []
s_to_n_DMS = []
sn_filter_2A3 = []
sn_filter_DMS = []
reactivity_2A3 = []
reactivity_DMS = []
reactivity_error_2A3 = []
reactivity_error_DMS = []
bpp_c1 = []
bpp_c2 = []
bpp_o = []
for ix in range(len(data_2A3)):
    s_2A3, s_DMS = data_2A3.iloc[ix], data_DMS.iloc[ix]

    id1, id2 = s_2A3["sequence_id"], s_DMS["sequence_id"]
    if id1 != id2:
        print("UNEXPECTED ERROR %s" % id1)
    
    ids.append(id1)

    # sequence encoded
    sequence_encoded = np.array([nucleotide_mapper[s] for s in s_2A3["sequence"]]).astype("int8")
    sequences.append(sequence_encoded)

    # metadata
    s_to_n_2A3.append(s_2A3["signal_to_noise"])
    s_to_n_DMS.append(s_DMS["signal_to_noise"])

    sn_filter_2A3.append(s_2A3["SN_filter"])
    sn_filter_DMS.append(s_DMS["SN_filter"])
    
    # reactivity and error
    reactivity_2A3_s = s_2A3[["reactivity_%04d" % (i+1) for i in range(206)]].values.astype("float32")
    reactivity_error_2A3_s = s_2A3[["reactivity_error_%04d" % (i+1) for i in range(206)]].values.astype("float32")

    reactivity_DMS_s = s_DMS[["reactivity_%04d" % (i+1) for i in range(206)]].values.astype("float32")
    reactivity_error_DMS_s = s_DMS[["reactivity_error_%04d" % (i+1) for i in range(206)]].values.astype("float32")

    reactivity_2A3.append(reactivity_2A3_s)
    reactivity_DMS.append(reactivity_DMS_s)
    reactivity_error_2A3.append(reactivity_error_2A3_s)
    reactivity_error_DMS.append(reactivity_error_DMS_s)

    # base pairing prob file loading into coordinate 1, coordinate 2, and output
    c1, c2, o = np.split(np.loadtxt(bpp_files[s_2A3["sequence_id"]]), 3, axis=1)
    c1, c2, o = c1.astype("int16")[:, 0], c2.astype("int16")[:, 0], o.astype("float16")[:, 0]
    bpp_c1.append(c1)
    bpp_c2.append(c2)
    bpp_o.append(o)

    if (ix+1) % 5000 == 0:
        print(ix, flush = True)

# train data h5torch file creation.
f = h5torch.File(train_data_output_file, "w")
f.register(np.array(ids), "central", dtype_save="bytes", dtype_load="str")

f.register(sequences, axis = 0, name = "sequences", mode = "vlen", dtype_save="int8", dtype_load="int64")
f.register(reactivity_2A3, axis = 0, name = "reactivity_2A3", mode = "vlen", dtype_save="float32", dtype_load="float32")
f.register(reactivity_DMS, axis = 0, name = "reactivity_DMS", mode = "vlen", dtype_save="float32", dtype_load="float32")
f.register(reactivity_error_2A3, axis = 0, name = "reactivity_error_2A3", mode = "vlen", dtype_save="float32", dtype_load="float32")
f.register(reactivity_error_DMS, axis = 0, name = "reactivity_error_DMS", mode = "vlen", dtype_save="float32", dtype_load="float32")

f.register(np.array(s_to_n_2A3), axis = 0, name = "s_to_n_2A3", mode = "N-D", dtype_save="float32", dtype_load="float32")
f.register(np.array(s_to_n_DMS), axis = 0, name = "s_to_n_DMS", mode = "N-D", dtype_save="float32", dtype_load="float32")
f.register(np.array(sn_filter_2A3), axis = 0, name = "sn_filter_2A3", mode = "N-D", dtype_save="bool", dtype_load="bool")
f.register(np.array(sn_filter_DMS), axis = 0, name = "sn_filter_DMS", mode = "N-D", dtype_save="bool", dtype_load="bool")

f.register(bpp_c1, axis = 0, name = "bpp_c1", mode = "vlen", dtype_save="int16", dtype_load="int16")
f.register(bpp_c2, axis = 0, name = "bpp_c2", mode = "vlen", dtype_save="int16", dtype_load="int16")
f.register(bpp_o, axis = 0, name = "bpp_o", mode = "vlen", dtype_save="float16", dtype_load="float32")

split = np.random.choice(np.arange(20), size = (len(sequences), ))
f.register(split, axis = 0, name = "split", dtype_save="int8", dtype_load="int8")

f.close()