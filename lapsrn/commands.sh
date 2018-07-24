

screen -Sdm "momentum" bash -c "python train.py -d 0 --solver Momentum --lr 1e-6 --monitor-path result/example_3 --use-bn"

screen -Sdm "adam-3" bash -c "python train.py -d 0 --solver Adam --lr 1e-3 --monitor-path result/example_4"
screen -Sdm "adam-4" bash -c "python train.py -d 0 --solver Adam --lr 1e-4 --monitor-path result/example_5"

screen -Sdm "adam-3-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-3 --monitor-path result/example_6 --use-bn"
screen -Sdm "adam-4-bn" bash -c "python train.py -d 1 --solver Adam --lr 1e-4 --monitor-path result/example_7 --use-bn"
