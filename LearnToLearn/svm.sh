#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons

nice python train_svms.py \
	-x ../data/features.pickle \
	-y ../data/labels.pickle \
	-o ../data/l2/w0 \
	-s w0
