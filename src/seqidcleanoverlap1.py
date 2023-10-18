"""
Selects all training sequences that pass the SN filter and that have only one occurence for both experiment types.
"""
import pandas as pd

# some consts
N=100000
WINDOW=10
R_COLS = ["reactivity_"+f"{num:04d}" for num in range(1,207)]
RE_COLS = ["reactivity_error_"+f"{num:04d}" for num in range(1,207)]
DATA_FOLDER = "/data/thomas_gaetan_share/kaggle/ribonanza/data/"

def main():
    print("Reading in training data...")
    train_data = pd.read_csv(DATA_FOLDER+"train_data.csv")
    print("Done!")
    print("Transform data...")
    # filter clean samples
    train_data_clean = train_data.loc[train_data.SN_filter==1,:]
    # split data based on experiment type
    train_data_2A3 = train_data_clean.loc[train_data_clean.experiment_type=="2A3_MaP",:]
    train_data_DMS = train_data_clean.loc[train_data_clean.experiment_type=="DMS_MaP",:]
    # get intersection wrt sequence_id
    seq_id_clean_overlap = set(train_data_2A3.sequence_id).intersection(train_data_DMS.sequence_id)
    # and subset data accordingly
    train_data_2A3_overlap = train_data_2A3.loc[train_data_2A3.sequence_id.isin(seq_id_clean_overlap),:]
    train_data_DMS_overlap = train_data_DMS.loc[train_data_DMS.sequence_id.isin(seq_id_clean_overlap),:]
    # now pick sequences that only occur ones in both experiment types
    seq_id_clean_overlap_1_2A3 = set(train_data_2A3_overlap.sequence_id.value_counts()[train_data_2A3_overlap.sequence_id.value_counts()==1].index)
    seq_id_clean_overlap_1_DMS = set(train_data_DMS_overlap.sequence_id.value_counts()[train_data_DMS_overlap.sequence_id.value_counts()==1].index)
    seq_id_clean_overlap_1 = seq_id_clean_overlap_1_DMS.intersection(seq_id_clean_overlap_1_2A3)
    train_data_2A3 = train_data_2A3_overlap.loc[train_data_2A3_overlap.sequence_id.isin(seq_id_clean_overlap_1),:]
    train_data_DMS = train_data_DMS_overlap.loc[train_data_DMS_overlap.sequence_id.isin(seq_id_clean_overlap_1),:]
    print("Done!")
    print(train_data_2A3.shape)
    print(train_data_DMS.shape)

if __name__ == "__main__":
    main()
