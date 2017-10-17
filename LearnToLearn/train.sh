#!/bin/bash
nice python train.py \
    -w0 ../data/w0/ \
    -w1 ../data/w1.pickle \
    -f ../data/features.pickle \
    -r ../data/runs/test_1 \
    --ckpt ../data/ckpts/test_1 \
    --write_every_n 100 \
    --validate_every_n 500 \
    --square_hinge

