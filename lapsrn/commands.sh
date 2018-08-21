
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
#screen -Sdm "momentum-27" bash -c "python train.py -d 1 -b 64 --solver Momentum --lr 1e-5 --monitor-path result/example_27 --share-type within-pyramid"


screen -Sdm "adam-28" bash -c "python train.py -d 1 -b 32 --solver Adam --lr 1e-4 --monitor-path result/example_28 --use-bn --decay-rate 0 --share-type across-pyramid"
#screen -Sdm "adam-29" bash -c "python train.py -d 2 -b 32 --solver Adam --lr 1e-4 --monitor-path result/example_29 --use-bn --decay-rate 0 --share-type within-pyramid"


screen -Sdm "adam-30" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-5 --monitor-path result/example_30 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-31" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-5 --monitor-path result/example_31 --decay-rate 0 --share-type within-pyramid"


#screen -Sdm "adam-32" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_32 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-33" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_33 --decay-rate 0 --share-type within-pyramid --skip-type ns"


#screen -Sdm "adam-34" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_34 --decay-rate 0 --share-type across-pyramid --R 4"
screen -Sdm "adam-35" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_35 --decay-rate 0 --share-type within-pyramid --R 4"


# OK
screen -Sdm "adam-36" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_36 --decay-rate 0 --share-type across-pyramid --S 2"
screen -Sdm "adam-37" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_37 --decay-rate 0 --share-type within-pyramid --S 2"


screen -Sdm "adam-38" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_38 --decay-rate 0 --share-type within-pyramid --S 2 --R 1"

screen -Sdm "adam-39" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_39 --decay-rate 0 --share-type within-pyramid --S 2 --R 1 --D 10"

screen -Sdm "adam-40" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_40 --decay-rate 0 --share-type within-pyramid --S 1 --R 1 --D 10"

screen -Sdm "adam-41" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_41 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-42" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_42 --decay-rate 0 --share-type within-pyramid"


screen -Sdm "momentum-43" bash -c "python train.py -d 1 -b 64 --solver Momentum --lr 1e-3 --monitor-path result/example_43 --share-type across-pyramid --R 3 --D 3"
screen -Sdm "momentum-44" bash -c "python train.py -d 2 -b 64 --solver Momentum --lr 1e-3 --monitor-path result/example_44 --share-type within-pyramid --R 3 --D 3"

# OK
screen -Sdm "adam-45" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_45 --decay-rate 0 --share-type within-pyramid --S 2 --R 1 --D 16"
screen -Sdm "adam-46" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_46 --decay-rate 0 --share-type within-pyramid --S 3 --R 1 --D 16"


screen -Sdm "adam-47" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_47 --decay-rate 0 --share-type across-pyramid"
screen -Sdm "adam-48" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_48 --decay-rate 0 --share-type within-pyramid"


screen -Sdm "adam-49" bash -c "python train.py -d 1 -b 32 --solver Adam --lr 1e-4 --monitor-path result/example_49 --decay-rate 0 --share-type across-pyramid --maps 128 --S 2"
screen -Sdm "adam-50" bash -c "python train.py -d 2 -b 32 --solver Adam --lr 1e-4 --monitor-path result/example_50 --decay-rate 0 --share-type within-pyramid --maps 128 --S 2"


# From here PIL is used
screen -Sdm "adam-51" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_51 --decay-rate 0 --share-type across-pyramid --S 2"
screen -Sdm "adam-52" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_52 --decay-rate 0 --share-type within-pyramid --S 2"

screen -Sdm "adam-53" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_53 --decay-rate 0 --share-type across-pyramid --S 3"
screen -Sdm "adam-54" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_54 --decay-rate 0 --share-type within-pyramid --S 3"

# Decay at every 10 epoch
screen -Sdm "adam-55" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_55 --decay-rate 0 --share-type across-pyramid --S 2"
screen -Sdm "adam-56" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_56 --decay-rate 0 --share-type within-pyramid --S 2"

screen -Sdm "adam-57" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_57 --decay-rate 0 --share-type across-pyramid --S 3"
screen -Sdm "adam-58" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_58 --decay-rate 0 --share-type within-pyramid --S 3"

# Decay at every 50 epoch
screen -Sdm "adam-59" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_59 --share-type across-pyramid --S 2"
screen -Sdm "adam-60" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_60 --share-type within-pyramid --S 2"

screen -Sdm "adam-61" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_61 --share-type across-pyramid --S 2 --R 1 --D 10"
screen -Sdm "adam-62" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_62 --share-type within-pyramid --S 2 --R 1 --D 10"

# Decay at every 50 epoch, not use General, and use random resize from [0.5, 0.75, 1.0]
screen -Sdm "adam-63" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_63 --share-type across-pyramid --S 2"
screen -Sdm "adam-64" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_64 --share-type within-pyramid --S 2"
screen -Sdm "adam-65" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_65 --share-type across-pyramid --S 3"
screen -Sdm "adam-66" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_66 --share-type within-pyramid --S 3"

# Decay at every 10 epoch and use sum instead of mean
screen -Sdm "adam-67" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_67 --share-type across-pyramid --S 2"
screen -Sdm "adam-68" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_68 --share-type within-pyramid --S 2"
screen -Sdm "adam-69" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_69 --share-type across-pyramid --S 3"
screen -Sdm "adam-70" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_70 --share-type within-pyramid --S 3"

# Decay at every 10 epoch and use sum instead of mean, BUI // -> /
screen -Sdm "adam-71" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_71 --share-type within-pyramid --S 2 --R 1 --D 10"
screen -Sdm "adam-72" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_72 --share-type within-pyramid --S 3 --R 1 --D 10"

# Decay at every 50 epoch, BUI // -> /, using all datasets
screen -Sdm "adam-73" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_73 --share-type across-pyramid --S 2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91 /home/kzky/nnabla_data/Set5 /home/kzky/nnabla_data/Set14 /home/kzky/nnabla_data/Manga109 /home/kzky/nnabla_data/Urban100"
screen -Sdm "adam-74" bash -c "python train.py -d 2 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_74 --share-type within-pyramid --S 2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91 /home/kzky/nnabla_data/Set5 /home/kzky/nnabla_data/Set14 /home/kzky/nnabla_data/Manga109 /home/kzky/nnabla_data/Urban100"

# Decay at every 50 epoch, BUI // -> /, using L2-loss
screen -Sdm "adam-75" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_75 --share-type across-pyramid --S 2 --loss l2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"
screen -Sdm "adam-76" bash -c "python train.py -d 0 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_76 --share-type within-pyramid --S 2 --loss l2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"

# Decay at every 50 epoch, BUI // -> /, using L1-loss
screen -Sdm "adam-77" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_77 --share-type across-pyramid --S 2 --loss l1 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"
screen -Sdm "adam-78" bash -c "python train.py -d 1 -b 64 --solver Adam --lr 1e-4 --monitor-path result/example_78 --share-type within-pyramid --S 2 --loss l1 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"


# Decay at every 50 epoch, BUI // -> /, using lr=1e-3, using L2-loss
screen -Sdm "adam-79" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_79 --share-type across-pyramid --S 2 --loss l2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"
screen -Sdm "adam-80" bash -c "python train.py -d 3 -b 64 --solver Adam --lr 1e-3 --monitor-path result/example_80 --share-type within-pyramid --S 2 --loss l2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/General100 /home/kzky/nnabla_data/T91"


# Align otriginal code
screen -Sdm "adam-81" bash -c "python train.py -d 2 -b 64 --solver Momentum --lr 5e-6 --monitor-path result/example_81 --share-type across-pyramid --S 2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/T91"
screen -Sdm "adam-82" bash -c "python train.py -d 3 -b 64 --solver Momentum --lr 5e-6 --monitor-path result/example_82 --share-type within-pyramid --S 2 --img-paths /home/kzky/nnabla_data/BSDS200 /home/kzky/nnabla_data/T91"


# Evaluate
## 51
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set14 --monitor-path result/example_51 --model-load-path result/example_51/param_149999.h5 --share-type across-pyramid --S 2
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set5 --monitor-path result/example_51 --model-load-path result/example_51/param_149999.h5 --share-type across-pyramid --S 2

## 52
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set14 --monitor-path result/example_52 --model-load-path result/example_52/param_149999.h5 --share-type within-pyramid --S 2
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set5 --monitor-path result/example_52 --model-load-path result/example_52/param_149999.h5 --share-type within-pyramid --S 2

## 53
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set14 --monitor-path result/example_53 --model-load-path result/example_53/param_149999.h5 --share-type across-pyramid --S 3
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set5 --monitor-path result/example_53 --model-load-path result/example_53/param_149999.h5 --share-type across-pyramid --S 3

## 54
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set14 --monitor-path result/example_54 --model-load-path result/example_54/param_149999.h5 --share-type within-pyramid --S 3
python evaluate.py -d 1 -c cudnn --valid-data-path ~/nnabla_data/Set5 --monitor-path result/example_54 --model-load-path result/example_54/param_149999.h5 --share-type within-pyramid --S 3
