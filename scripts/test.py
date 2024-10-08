import subprocess

cmd = r'python main.py --project Segment-Drive ' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--test '.split()

subprocess.run(cmd)
