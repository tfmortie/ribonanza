"""
Python script to merge two submission files, corresponding to DMS and 2A3, respectively, to one final kaggle-submission-ready file.
"""
import sys

import pandas as pd

def merge(args):
    # get filenames
    submission_dms = args[1]
    submission_2a3 = args[2]
    out = args[3]
    # read in csv files
    submission_dms_df = pd.read_csv(submission_dms)
    submission_2a3_df = pd.read_csv(submission_2a3)
    final_csv = pd.DataFrame({"id": submission_dms_df.iloc[:,0], "reactivity_DMS_MaP": submission_dms_df.Pred, "reactivity_2A3_MaP": submission_2a3_df.Pred})
    # save to 
    final_csv.to_csv("./"+out, index=False)
    
if __name__ == "__main__":
    merge(sys.argv)
