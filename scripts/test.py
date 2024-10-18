import subprocess

cmds = dict()

cmds['UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--test '.split()

cmds['Which-Way-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--test '.split()

cmds['WayAttention-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m WayAttention-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--test '.split()

cmds['SWA-UNet'] = r'python main.py --project Segment-Drive --author LpoutYoumu' \
    '-m SWA-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--test '.split()

subprocess.run(cmds['UNet'])
