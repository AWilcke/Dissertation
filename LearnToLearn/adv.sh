#!/bin/bash

python adv.py \
    -w0 ../data/w0 \
    -w1 ../data/w1.pickle \
    -f ../data/features.pickle \
    -r ../data/runs/test1 \
    --ckpt ../data/ckpts/test1 \
    --optimiser rmsprop \
    --lr 0.0005 \
    --write_every_n 100 \
    --validate_every_n 500
