import subprocess

cmds = dict()

cmds['UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '-lr 3e-5 -e 200 -b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--augment_boost ' \
    '--train ' \
    '--save_every_n_epoch 40'.split()

cmds['Which-Way-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '-lr 3e-5 -e 200 -b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--augment_boost ' \
    '--train ' \
    '--save_every_n_epoch 40'.split()

cmds['WayAttention-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m WayAttention-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '-lr 3e-5 -e 200 -b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--augment_boost ' \
    '--train ' \
    '--save_every_n_epoch 40'.split()

cmds['SWA-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m SWA-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '-lr 3e-5 -e 200 -b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--augment_boost ' \
    '--train ' \
    '--save_every_n_epoch 40'.split()

subprocess.run(cmds['UNet'])
