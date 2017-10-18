#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

export CUDA_VISIBLE_DEVICES=2

nice python train.py \
    -w0 ../data/l2/w0/ \
    -w1 ../data/l2/w1.pickle \
    -f ../data/features.pickle \
    -r ../data/runs/l2_squared \
    --ckpt ../data/ckpts/l2_squared \
    --write_every_n 100 \
    --validate_every_n 500 \
    --steps 33 66 \
    --n_models_to_keep 1 \
    --square_hinge
