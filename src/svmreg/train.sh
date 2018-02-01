#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

export CUDA_VISIBLE_DEVICES=0

name=rms

nice python train.py \
    -w0 ../data/l2/w0/ \
    -w1 ../data/l2/w1.pickle \
    -f ../data/features.pickle \
    -l ../data/labels.pickle \
    -r ../data/runs/basic/$name \
    --ckpt ../data/ckpts/basic/$name \
    --write_every_n 100 \
    --validate_every_n 500 \
    --classif_every_n 1000 \
    --n_models_to_keep 1 \
    --optimiser rmsprop
