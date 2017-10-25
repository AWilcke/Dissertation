#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

export CUDA_VISIBLE_DEVICES=1

nice python train.py \
    -w0 ../data/l2/w0/ \
    -w1 ../data/l2/w1.pickle \
    -f ../data/features.pickle \
    -r ../data/runs/l2_05\
    --ckpt ../data/ckpts/l2_05 \
    --write_every_n 100 \
    --validate_every_n 500 \
    --steps 33 66 \
    --n_models_to_keep 1 \
    --lr_weight 0.5
