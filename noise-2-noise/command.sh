# TRAIN (noisy)
# UNet
screen -Sdm "unet-gaussian" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_gaussian_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist gaussian --noise-level 50"

screen -Sdm "unet-poisson" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_poisson_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist poisson --noise-level 50"

screen -Sdm "unet-bernoulli" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_bernoulli_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist bernoulli --noise-level 0.95 --n-replica 16"

screen -Sdm "unet-impulse" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_impulse_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir --net unet --noise-dist impulse --noise-level 0.95 --loss l0 --n-replica 16"


# RED30
screen -Sdm "red30-gaussian" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/red30_gaussian_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net red30 --noise-dist gaussian --noise-level 50"


# TRAIN (clean)
# UNet
screen -Sdm "unet-gaussian-clean" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_gaussian_clean --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist gaussian --noise-level 50 --n-replica 1"

screen -Sdm "unet-poisson-clean" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_poisson_clean --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist poisson --noise-level 50 --n-replica 1"

screen -Sdm "unet-bernoulli-clean" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_bernoulli_clean --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist bernoulli --noise-level 0.95 --n-replica 1"

screen -Sdm "unet-impulse-clean" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_impulse_clean --train-data-path /data/datasets/imagenet/val_data/tmpdir --net unet --noise-dist impulse --noise-level 0.95 --loss l0 --n-replica 1"


# RED30
screen -Sdm "red30-gaussian" bash -c "python train.py -d 1 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/red30_gaussian_clean --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net red30 --noise-dist gaussian --noise-level 50"


# EVAL
python evaluate.py -d 2 -c "cudnn" --noise-dist gaussian --noise-level 25 --monitor-path result/red30_gaussian_2 --net red30 --model-load-path result/red30_gaussian_2/param_140625.h5
python evaluate.py -d 2 -c "cudnn" --noise-dist gaussian --noise-level 25 --monitor-path result/unet_gaussian_2 --net unet --model-load-path result/unet_gaussian_2/param_140625.h5
python evaluate.py -d 2 -c "cudnn" --noise-dist poisson --noise-level 30 --monitor-path result/unet_poisson_2 --net unet --model-load-path result/unet_poisson_2/param_140625.h5
python evaluate.py -d 2 -c "cudnn" --noise-dist bernoulli --noise-level 0.5 --monitor-path result/unet_bernoulli_2 --net unet --model-load-path result/unet_bernoulli_2/param_140625.h5
python evaluate.py -d 2 -c "cudnn" --noise-dist impulse --noise-level 0.5 --monitor-path result/unet_impulse_2 --net unet --model-load-path result/unet_impulse_2/param_140625.h5

