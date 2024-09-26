@echo off
python main.py --project "Segment-Drive" ^
    -m "SWA-UNet" --n_channels 1 --n_classes 1 --classes "vein" ^
    -lr 3e-5 -e 40 -b 1 ^
    --data_dir "G:/AI/Data/" --dataset DRIVE ^
    --gpu --augment_boost ^
    --train ^
    --save_every_n_epoch 1
