
# UNet
screen -Sdm "unet-gaussian" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_gaussian --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist gaussian --noise-level 50"

screen -Sdm "unet-poisson" bash -c "python train.py -d 0 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_poisson --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist poisson --noise-level 50  --n-replica 8"

screen -Sdm "unet-bernoulli" bash -c "python train.py -d 1 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_bernoulli --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net unet --noise-dist bernoulli --noise-level 0.95"

screen -Sdm "unet-impulse" bash -c "python train.py -d 1 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/unet_impulse --train-data-path /data/datasets/imagenet/val_data/tmpdir --net unet --noise-dist impulse --noise-level 0.95 --loss l0"


# RED30
screen -Sdm "red30-gaussian" bash -c "python train.py -d 2 -c cudnn -b 4 --max-iter 156250 --save-interval 15625 --monitor-path ./result/red30_gaussian --train-data-path /data/datasets/imagenet/val_data/tmpdir/ --net red30 --noise-dist gaussian --noise-level 50"

