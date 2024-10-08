import subprocess

cmd = r'python main.py --project Segment-Drive ' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '--gpu ' \
    '--predict ' \
    '--load output/best_model.pth ' \
    '--input D:/Data/Datasets/DRIVE/test/images'.split()

subprocess.run(cmd)