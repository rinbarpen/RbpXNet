@echo off
python main.py --project "Segment-Drive" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "vein" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset DRIVE ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20

python main.py --project "Segment-ISIC2018" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "x" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset ISIC2018 ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20

python main.py --project "Segment-ISIC2017" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "x" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset ISIC2017 ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20

python main.py --project "Segment-FIVES" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "vein" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset FIVES ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20

python main.py --project "Segment-PLOYP2021" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "lung" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset PLOYP2021 ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20

python main.py --project "Segment-BOWL2018" ^
    -m "ML-UNet" --n_channels 1 --n_classes 1 --classes "kernel" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset BOWL2018 ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20
python main.py --project "Segment-BOWL2018" ^
    -m "Which-Way-UNet" --n_channels 1 --n_classes 1 --classes "kernel" ^
    -lr 3e-5 -e 200 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset BOWL2018 ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 20
