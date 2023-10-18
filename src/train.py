"""
"""
import pandas as pd

# some consts
N=100000
WINDOW=10
R_COLS = ["reactivity_"+f"{num:04d}" for num in range(1,207)]
RE_COLS = ["reactivity_error_"+f"{num:04d}" for num in range(1,207)]
DATA_FOLDER = "/data/thomas_gaetan_share/kaggle/ribonanza/data/"

