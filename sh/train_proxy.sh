#!/bin/bash

python simclr.py \
    --arch resnet10 \
    --dataset cifar10 \
    --device 0 \
    --seed 0