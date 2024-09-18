import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops.misc import interpolate

SplitType = Literal['train', 'valid', 'test']

class DriveDataset(Dataset):
    # 'train' 'valid' 'test'
    def __init__(self, dirpath: str, split: SplitType='train', tv_ratio=0.2, transforms=None, use_rgb=False):
        self.dir = Path(dirpath)
        self.transforms = transforms
        self.use_rgb = use_rgb

        if split == 'train':
            self.image_dir = self.dir / 'training' / 'images'
            self.mask_dir = self.dir / 'training' / '1st_manual'
        elif split == 'valid':
            self.image_dir = self.dir / 'training' / 'images'
            self.mask_dir = self.dir / 'training' / '1st_manual'
        elif split == 'test':
            self.image_dir = self.dir / 'test' / 'images'
            self.mask_dir = self.dir / 'test' / '1st_manual'

        self.images = [self.image_dir / image for image in self.image_dir.glob('*.tif')]
        self.masks = [self.mask_dir / mask for mask in self.mask_dir.glob('*.gif')]

        if split == 'train':
            self.images = self.images[:int(len(self.images) * (1 - tv_ratio))]
            self.masks = self.masks[:int(len(self.masks) * (1 - tv_ratio))]
        elif split == 'valid':
            self.images = self.images[int(len(self.images) * (1 - tv_ratio)):]
            self.masks = self.masks[int(len(self.masks) * (1 - tv_ratio)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx], self.masks[idx]

        if self.use_rgb:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # image = cv2.imread(str(image_path))
        # mask = cv2.imread(str(mask_path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_np = np.array(mask)
        if mask_np.max() > 1:
            mask_np = mask_np / 255
        mask = Image.fromarray(mask_np, mode='L')

        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)

        return image, mask

    @staticmethod
    def get_train_valid_and_test(drive_dir, tv_ratio=0.2, transforms=None):
        train_set = DriveDataset(drive_dir, 'train', tv_ratio, transforms=transforms[0])
        valid_set = DriveDataset(drive_dir, 'valid', tv_ratio, transforms=transforms[0])
        test_set  = DriveDataset(drive_dir, 'test',  tv_ratio, transforms=transforms[1])
        return train_set, valid_set, test_set


class DriveDataset2(Dataset):
    # 'train' 'valid' 'test'
    def __init__(self, dirpath: str, split: SplitType='train', tv_ratio=0.2, transforms=None, use_rgb=False):
        self.dir = Path(dirpath)
        self.transforms = transforms
        self.use_rgb = use_rgb

        if split == 'train':
            self.image_dir = self.dir / 'training' / 'images'
            self.mask_dir = self.dir / 'training' / '1st_manual'
        elif split == 'valid':
            self.image_dir = self.dir / 'training' / 'images'
            self.mask_dir = self.dir / 'training' / '1st_manual'
        elif split == 'test':
            self.image_dir = self.dir / 'test' / 'images'
            self.mask_dir = self.dir / 'test' / '1st_manual'

        self.images = [self.image_dir / image for image in self.image_dir.glob('*.tif')]
        self.masks = [self.mask_dir / mask for mask in self.mask_dir.glob('*.gif')]

        if split == 'train':
            self.images = self.images[:int(len(self.images) * (1 - tv_ratio))]
            self.masks = self.masks[:int(len(self.masks) * (1 - tv_ratio))]
        elif split == 'valid':
            self.images = self.images[int(len(self.images) * (1 - tv_ratio)):]
            self.masks = self.masks[int(len(self.masks) * (1 - tv_ratio)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import cv2
        image_path, mask_path = self.images[idx], self.masks[idx]

        image = cv2.imread(str(image_path))  # (H, W, C)
        mask = cv2.imread(str(mask_path))
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.max() > 1:
            mask = mask / 255

        # if self.transforms:
        #     image, mask = self.transforms(image), self.transforms(mask)
        flip_code = random.choice([-1, 0, 1, 2])
        if flip_code != 2:
            image = self.augment(image, flip_code)
            mask = self.augment(mask, flip_code)

        image = image.reshape(1, image.shape[0], image.shape[1])
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])
        return image, mask

    @staticmethod
    def get_train_valid_and_test(drive_dir, tv_ratio=0.2, transforms=None):
        train_set = DriveDataset(drive_dir, 'train', tv_ratio, transforms=transforms[0])
        valid_set = DriveDataset(drive_dir, 'valid', tv_ratio, transforms=transforms[0])
        test_set  = DriveDataset(drive_dir, 'test',  tv_ratio, transforms=transforms[1])
        return train_set, valid_set, test_set
