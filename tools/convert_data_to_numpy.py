import logging
import os.path
from pathlib import Path

import numpy as np
from PIL import Image

from utils.utils import file_prefix_name, create_dirs


def convert_image_to_numpy(filepath: str | Path, is_rgb=True):
    if is_rgb:
        image = Image.open(filepath).convert('RGB')
        data = np.array(image, dtype=np.float32).transpose(2, 0, 1) # (C, H, W)
    else:
        image = Image.open(filepath).convert('L')
        data = np.array(image, dtype=np.float32) # (H, W)

    data_filename = file_prefix_name(filepath) + '.npz'

    data_dir = 'output/data'
    create_dirs(data_dir)
    data_filepath = os.path.join(data_dir, data_filename)
    np.save(data_filepath, data)

    logging.info(f'Convert {filepath} to {data_filepath}')

def convert_images_to_numpy(root: str | Path, is_rgb=True):
    if isinstance(root, str):
        for filenames in os.listdir(root):
            for filename in filenames:
                convert_image_to_numpy(os.path.join(root, filename), is_rgb)
    elif isinstance(root, Path):
        for filename in root.iterdir():
            convert_image_to_numpy(root / filename, is_rgb)
