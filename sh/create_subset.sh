#!/bin/bash

for frac in 0.2 0.4 0.6 0.8
do
    python create_subset.py \
        --dataset cifar100 \
        --device 0 \
        --subset-fraction $frac \
        --net-path 'ckpt/2023-11-30 01:24:35.890-cifar100-resnet10-seed0-399-net.pt' \
        --critic-path 'ckpt/2023-11-30 01:24:35.890-cifar100-resnet10-seed0-399-critic.pt' \
        --subset-path subset_indices \
        --seed 0
done