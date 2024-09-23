#!/usr/bin bash
conda activate py310
python main.py --project 'Segment-Drive' \
  -m UNet --in_channels 3 --n_classes 1 --classes "vein" \
  -b 1 \
  --data_dir 'D:/Data/Datasets/' --dataset 'DRIVE' \
  --gpu \
  --test
