import glob
import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import *


class DatasetTemplate(Dataset):
    def __init__(self):
        super(DatasetTemplate, self).__init__()

    def __getitem__(self, idx):
        return ()


class DatasetImageImageTemplate(DatasetTemplate):
    def __init__(self, original_path, target_path, original_suffix, target_suffix, transformer=None, **kwargs):
        super(DatasetImageImageTemplate, self).__init__()

        self.transformer = transformer

        self.original_image_src = [path for path in glob.glob(original_path) if os.path.splitext(path)[1].endswith(original_suffix)]
        self.target_image_src = [path for path in glob.glob(target_path) if os.path.splitext(path)[1].endswith(target_suffix)]
        self.config = kwargs

    def __getitem__(self, idx):
        image_path, mask_path = self.original_image_src[idx], self.target_image_src[idx]

        if self.config['original']['rgb']:
            image = Image.open(image_path).convert('RGB')
            # image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(image_path).convert('L')
            # image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

        if self.config['target']['rgb']:
            mask = Image.open(mask_path).convert('RGB')
            # mask = cv2.imread(mask_path, cv2.COLOR_BGR2RGB)
        else:
            mask = Image.open(mask_path).convert('L')
            # mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)

        if self.transformer:
            image, mask = self.transformer[0](image), self.transformer[1](mask)

        return image, mask

