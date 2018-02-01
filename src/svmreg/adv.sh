#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

export CUDA_VISIBLE_DEVICES=2

name=gp_a10_4critic_rms_25drop

python adv.py \
    -w0 ../data/l2/w0 \
    -w1 ../data/l2/w1.pickle \
    -f ../data/features.pickle \
    -l ../data/labels.pickle \
    -r ../data/runs/$name \
    --ckpt ../data/ckpts/$name \
    --type wgan \
    --gp \
    --optimiser_G rmsprop \
    --optimiser_C rmsprop \
    --lr_C 5e-5\
    --lr_G 5e-5 \
    --critic_iters 5 \
    --alpha 10 \
    --gen_name standard \
    --critic_name critic4 \
    --dropout 0.25
