import subprocess

cmds = dict()

cmds['UNet'] = r'python main.py --project Segment-Drive ' \
    '-m UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '--gpu ' \
    '--predict ' \
    '--load output/best_model.pth ' \
    '--input D:/Data/Datasets/DRIVE/test/images'.split()

cmds['Which-Way-UNet'] = r'python main.py --project Segment-Drive ' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '--gpu ' \
    '--predict ' \
    '--load output/best_model.pth ' \
    '--input D:/Data/Datasets/DRIVE/test/images'.split()

cmds['WayAttention-UNet'] = r'python main.py --project Segment-Drive ' \
    '-m WayAttention-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '--gpu ' \
    '--predict ' \
    '--load output/best_model.pth ' \
    '--input D:/Data/Datasets/DRIVE/test/images'.split()

cmds['SWA-UNet'] = r'python main.py --project Segment-Drive ' \
    '-m SWA-UNet --n_channels 1 --n_classes 1 ' \
    '--classes vein ' \
    '--gpu ' \
    '--predict ' \
    '--load output/best_model.pth ' \
    '--input D:/Data/Datasets/DRIVE/test/images'.split()

subprocess.run(cmds['UNet'])
