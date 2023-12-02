#!/bin/bash

python create_subset.py \
    --dataset cifar100 \
    --device 0 \
    --subset-fractions 0.2 0.4 0.6 0.8 \
    --net-path 'ckpt/2023-11-30 01:32:57.421-tiny_imagenet-resnet10-seed0-399-net.pt' \
    --critic-path 'ckpt/2023-11-30 01:32:57.421-tiny_imagenet-resnet10-seed0-399-critic.pt' \
    --subset-path subset_indices \
    --proxy-img-size 64 \
    --proxy-dataset tiny_imagenet \
    --proxy-arch resnet10 \
    --seed 0