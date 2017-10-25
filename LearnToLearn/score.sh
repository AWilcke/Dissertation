#!/bin/bash

export PATH=/home/s1452854/miniconda3/bin:$PATH
source activate hons
export CUDA_VISIBLE_DEVICES=0

name='l2_squared'
x='../data/features.pickle'
y='../data/imagelabels.mat'
o='../results'
ckpt='../data/ckpts/l2_squared/100.ckpt'
val_path='../data/l2/w0/val'
c=1
loss='squared_hinge'

nice python scoring.py \
	-x $x \
	-y $y \
	-o "${o}/${name}" \
	--ckpt $ckpt \
	--val_path $val_path \
	-C $c \
	--loss $loss

nice python scoring.py \
	-x $x \
	-y $y \
	-o "${o}/${name}_refit_${c}" \
	--ckpt $ckpt \
	--val_path $val_path \
	-C $c \
	--loss $loss \
	--refit

nice python scoring.py \
	-x $x \
	-y $y \
	-o "${o}/${name}_w0" \
	--ckpt $ckpt \
	--val_path $val_path \
	-C $c \
	--loss $loss \
	--usew0
