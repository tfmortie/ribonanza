#!/bin/bash

# MTM model on high-quality sequences that occur once in both DMS and 2A3 experiments
python train.py -nepochs 73 -d train_data_DMS_1.csv -out model_dms_1_2310_457 -v 2
python train.py -nepochs 73 -d train_data_2A3_1.csv -out model_2A3_1_2310_457 -v 2