#!/bin/bash

# MTM model on high-quality sequences that occur once in both DMS and 2A3 experiments
python test.py --d test_sequences.csv -mp model_2A3_1_2310_457.ckpt -embs 3 -mtmh1 64 -mtmh2 128 -seqlen 457 -bs 512 --out submission_2A3_1_2310_457
python test.py --d test_sequences.csv -mp model_dms_1_2310_457.ckpt -embs 3 -mtmh1 64 -mtmh2 128 -seqlen 457 -bs 512 --out submission_DMS_1_2310_457