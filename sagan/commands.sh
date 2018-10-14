#!/bin/bash

# 000: default
mpirun -n 4 python train_with_mgpu.py -c "cudnn" -b 16 -a 4 -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_000 --max-iter 450000 --save-interval 10000

