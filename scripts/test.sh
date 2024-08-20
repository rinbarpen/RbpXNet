#!/usr/bin bash
conda activate py310
python main.py --proj 'Segment-ISIC2017' -m UNet --in_channels 3 --n_classes 1 -lr 1e-3 -e 10 -b 1 --data_dir 'D:/Data/Datasets/' --data 'ISIC2017' --test
