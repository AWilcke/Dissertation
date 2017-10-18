#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

nice python train_svms.py \
	-x ../data/features.pickle \
	-y ../data/imagelabels.mat \
	-o ../data/l2 \
	-s w1
