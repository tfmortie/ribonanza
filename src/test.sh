#!/bin/bash

# MTM model on high-quality sequences that occur once in both DMS and 2A3 experiments
#python test.py -m conv -d test_sequences.csv -mp model_2A3_1_2310_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_2A3_1_2310_457
#python test.py -m conv -d test_sequences.csv -mp model_dms_1_2310_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_DMS_1_2310_457

# MTM model on high-quality sequences that occur in both DMS and 2A3 experiments
#python test.py -m conv -d test_sequences.csv -mp model_2A3_1N_1811_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_2A3_1N_1811_457
#python test.py -m conv -d test_sequences.csv -mp model_dms_1N_1811_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_DMS_1N_1811_457

# MTM model on high-quality sequences that occur in both DMS and 2A3 experiments with weighted loss
python test.py -m conv -d test_sequences.csv -mp model_2A3_1N_w_1911_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_2A3_1N_w_1911_457
python test.py -m conv -d test_sequences.csv -mp model_dms_1N_w_1911_457.ckpt -embs 128 -nl 10 -seqlen 457 -bs 512 -dev 1 --out submission_DMS_1N_w_1911_457
