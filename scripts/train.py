import subprocess

# cmd = r'python main.py --project Segment-Drive ' \
#     '-m UNet --n_channels 1 --n_classes 1 ' \
#     '--classes "vein" ' \
#     '-lr 3e-5 -e 40 -b 1 ' \
#     '--data_dir D:/Data/Datasets/ ' \
#     '--dataset DRIVE ' \
#     '--gpu ' \
#     '--augment_boost ' \
#     '--train ' \
#     '--save_every_n_epoch 1'.split()

cmd = r'python main.py --project Segment-LM-Drive ' \
    '-m Which-Way-UNet --n_channels 1 --n_classes 1 ' \
    '--classes "vein" ' \
    '-lr 3e-5 -e 200 -b 1 ' \
    '--data_dir D:/Data/Datasets/ ' \
    '--dataset DRIVE ' \
    '--gpu ' \
    '--augment_boost ' \
    '--train ' \
    '--save_every_n_epoch 40'.split()

subprocess.run(cmd)
