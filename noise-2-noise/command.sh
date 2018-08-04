
# UNet
screen -Sdm "unet-gaussian" bash -c "python train.py -d 2 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_gaussian_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist gaussian --noise-level 50"

screen -Sdm "unet-poisson" bash -c "python train.py -d 2 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_poisson_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist poisson --noise-level 50"

screen -Sdm "unet-bernoulli" bash -c "python train.py -d 2 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_bernoulli_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist bernoulli --noise-level 0.95 --n-replica 16"

screen -Sdm "unet-impulse" bash -c "python train.py -d 2 -c cudnn -b 1 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_impulse_2 --train-data-path /data/datasets/imagenet/val_data/tmpdir --net unet --noise-dist impulse --noise-level 0.95 --loss l0 --n-replica 16"


# RED30
screen -Sdm "red30-gaussian" bash -c "python train.py -d 2 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/red30_gaussian --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net red30 --noise-dist gaussian --noise-level 50"

