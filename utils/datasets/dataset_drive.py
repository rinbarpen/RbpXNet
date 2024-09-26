from pathlib import Path
import random
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, dirpath: str, split: Literal['train', 'test']='train'):
        self.dir = Path(dirpath)

        if split == 'train':
            self.image_dir = self.dir / 'training' / 'images'
            self.mask_dir = self.dir / 'training' / '1st_manual'
        elif split == 'test':
            self.image_dir = self.dir / 'test' / 'images'
            self.mask_dir = self.dir / 'test' / '1st_manual'

        self.images = [self.image_dir / image for image in self.image_dir.glob('*.png')]
        self.masks = [self.mask_dir / mask for mask in self.mask_dir.glob('*.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx], self.masks[idx]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if mask.max() > 1:
            mask = mask / 255

        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            mask = self.augment(mask, flipCode)

        image = image.reshape(1, image.shape[0], image.shape[1])
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])
        
        return image, mask

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    @staticmethod
    def get_train_valid_and_test(drive_dir):
        train_set = DriveDataset(drive_dir, 'train')
        valid_set = None
        test_set  = DriveDataset(drive_dir, 'test')
        return train_set, valid_set, test_set
