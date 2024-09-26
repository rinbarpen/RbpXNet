@echo off
python main.py --project "Segment-Drive" ^
  -m UNet --n_channels 1 --n_classes 1 --classes "vein" ^
  -b 1 ^
  --data_dir "G:/AI/Data/" --dataset "DRIVE" ^
  --gpu ^
  --test
