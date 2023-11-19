#!/bin/bash

# MTM model on high-quality sequences that occur once in both DMS and 2A3 experiments
# python train.py -nepochs 73 -d train_data_DMS_1.csv -out model_dms_1_2310_457 -v 2
# python train.py -nepochs 73 -d train_data_2A3_1.csv -out model_2A3_1_2310_457 -v 2

# MTM model on high-quality sequences that occur in both DMS and 2A3 experiments
#python train.py -m conv -d train_data_DMS_1.csv train_data_DMS_N.csv -out model_dms_1N_1811_457 -v 2 -embs 128 -nl 10 -bs 256 -lr 0.0003 -dev 1 -ne 20
#python train.py -m conv -d train_data_2A3_1.csv train_data_2A3_N.csv -out model_2A3_1N_1811_457 -v 2 -embs 128 -nl 10 -bs 256 -lr 0.0003 -dev 1 -ne 20

# MTM model on high-quality sequences that occur in both DMS and 2A3 experiments with weighted MSE
python train.py -m conv -d train_data_DMS_1.csv train_data_DMS_N.csv -out model_dms_1N_w_1911_457 -v 2 -lo wmmse -embs 128 -nl 10 -bs 256 -lr 0.0003 -dev 1 -ne 20
python train.py -m conv -d train_data_2A3_1.csv train_data_2A3_N.csv -out model_2A3_1N_w_1911_457 -v 2 -lo wmmse -embs 128 -nl 10 -bs 256 -lr 0.0003 -dev 1 -ne 20