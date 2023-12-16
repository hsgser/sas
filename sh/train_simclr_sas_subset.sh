#!/bin/bash

DATASET=cifar100
for frac in 0.2 0.4 0.6 0.8
do
    SUBSET_PATH=subset_indices/${DATASET}-${frac}-sas-indices-vit-mae-base-imagenet1k.pkl
    python simclr.py \
        --arch resnet50 \
        --dataset $DATASET \
        --subset-indices $SUBSET_PATH \
        --device 0 \
        --seed 0
done