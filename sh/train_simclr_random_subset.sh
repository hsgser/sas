#!/bin/bash

for frac in 0.2 0.4 0.6 0.8
do
    python simclr.py \
        --arch resnet50 \
        --dataset cifar100 \
        --random-subset \
        --subset-fraction $frac \
        --device-ids 2 3 \
        --seed 0
done