#!/bin/bash

python create_subset_vit_mae.py \
    --dataset cifar100 \
    --device 0 \
    --subset-fractions 0.2 0.4 0.6 0.8 \
    --subset-path subset_indices \
    --proxy-img-size 224 \
    --proxy-dataset imagenet1k \
    --proxy-arch vit-mae-base \
    --proxy-pretrain facebook/vit-mae-base \
    --seed 0