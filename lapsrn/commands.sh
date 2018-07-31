
# nnabla v1.0.1
screen -Sdm "momentum" bash -c "python train.py -d 0 --solver Momentum --lr 1e-6 --monitor-path result/example_3 --use-bn"

screen -Sdm "adam-3" bash -c "python train.py -d 0 --solver Adam --lr 1e-3 --monitor-path result/example_4"

screen -Sdm "adam-4" bash -c "python train.py -d 2 --solver Adam --lr 1e-4 --monitor-path result/example_5"

screen -Sdm "adam-3-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-3 --monitor-path result/example_6 --use-bn"
screen -Sdm "adam-4-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-4 --monitor-path result/example_7 --use-bn"

# nnabla v0.9.9
screen -Sdm "momentum" bash -c "python train.py -d 0 --solver Momentum --lr 1e-6 --monitor-path result/example_8 --use-bn"

screen -Sdm "adam-3" bash -c "python train.py -d 0 --solver Adam --lr 1e-3 --monitor-path result/example_9"

screen -Sdm "adam-4" bash -c "python train.py -d 2 --solver Adam --lr 1e-4 --monitor-path result/example_10"

screen -Sdm "adam-3-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-3 --monitor-path result/example_11 --use-bn"

screen -Sdm "adam-4-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-4 --monitor-path result/example_12 --use-bn"


screen -Sdm "adam-4" bash -c "python train.py -d 1 --solver Adam --lr 1e-4 --monitor-path result/example_13 --use-shared"
screen -Sdm "adam-4" bash -c "python train.py -d 2 --solver Adam --lr 1e-4 --monitor-path result/example_14"

screen -Sdm "momentum" bash -c "python train.py -d 1 --solver Momentum --lr 1e-6 --monitor-path result/example_15 --use-shared  --use-bn"
screen -Sdm "momentum" bash -c "python train.py -d 2 --solver Momentum --lr 1e-6 --monitor-path result/example_16  --use-bn"

# screen -Sdm "adam-4" bash -c "python train.py -d 1 -S 2 --solver Adam --lr 1e-4 --monitor-path result/example_17 --use-shared"
# screen -Sdm "adam-4" bash -c "python train.py -d 2 -S 2 --solver Adam --lr 1e-4 --monitor-path result/example_18"

# screen -Sdm "momentum" bash -c "python train.py -d 1 -S 2 --solver Momentum --lr 1e-6 --monitor-path result/example_19 --use-shared  --use-bn"
# screen -Sdm "momentum" bash -c "python train.py -d 2 -S 2 --solver Momentum --lr 1e-6 --monitor-path result/example_20  --use-bn"

screen -Sdm "adam-21" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_21 --use-shared --decay-rate 0"
screen -Sdm "adam-22" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_22 --decay-rate 0"

#screen -Sdm "adam-23" bash -c "python train.py -d 0 -b 64 --solver Momentum --lr 1e-5 --monitor-path result/example_23 --use-shared"

screen -Sdm "adam-24" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_24 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-25" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_25 --decay-rate 0 --share-type within-pyramid"


screen -Sdm "momentum-26" bash -c "python train.py -d 0 -b 64 --solver Momentum --lr 1e-5 --monitor-path result/example_26 --share-type across-pyramid"
screen -Sdm "momentum-27" bash -c "python train.py -d 1 -b 64 --solver Momentum --lr 1e-5 --monitor-path result/example_27 --share-type within-pyramid"

screen -Sdm "adam-28" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_28 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-29" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_29 --decay-rate 0 --share-type within-pyramid"