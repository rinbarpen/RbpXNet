#!/usr/bin bash
conda activate py310
python main.py --proj 'Segment-Drive' \
  -m UNet --in_channels 3 --n_classes 1 --classes "vein" \ 
  -lr 3e-6 -e 40 -b 1 \
  --data_dir 'D:/Data/Datasets/' --dataset DRIVE \
  --gpu --augment_boost \
  --train \
  --save_every_n_epoch 1 \
