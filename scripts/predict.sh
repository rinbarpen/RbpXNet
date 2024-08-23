#!/usr/bin bash
conda activate py310
python main.py --project 'Segment-Drive' -m UNet --in_channels 3 --n_classes 1 --gpu --predict --load 'output/best_model.pth' --input 'D:/Data/Datasets/DRIVE/test/images/01_test.tif'
