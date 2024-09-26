@echo off

python main.py --project "Segment-Drive" -m "UNet" --n_channels 1 --n_classes 1 --classes "vein" --gpu --predict --load "output/best_model.pth" --input "G:/AI/Data/DRIVE/test/images"  
