#!/bin/bash

DATASET=cifar100
for frac in 0.2 0.4 0.6 0.8
do
    SUBSET_PATH=subset_indices/${DATASET}-${frac}-sas-indices.pkl
    python simclr.py \
        --arch resnet50 \
        --dataset $DATASET \
        --subset-indices $SUBSET_PATH \
        --device-ids 4 5 \
        --seed 0
done