#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

export CUDA_VISIBLE_DEVICES=0

nice python adv.py \
    -w0 ../data/l2/w0 \
    -w1 ../data/l2/w1.pickle \
    -f ../data/features.pickle \
    -r ../data/runs/adv1 \
    --ckpt ../data/ckpts/adv1 \
    --optimiser rmsprop \
    --lr 0.0005 \
