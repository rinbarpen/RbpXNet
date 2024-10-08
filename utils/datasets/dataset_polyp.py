import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PolypGen2021Dataset(Dataset):
    def __init__(self, polyp_dir, split: Literal['train', 'valid', 'test'], valid_ratio=0.2, transforms=None):
        super(PolypGen2021Dataset, self).__init__()
        self.polyp_dir = Path(polyp_dir)
        self.transforms = transforms
        self.valid_ratio = valid_ratio

        if split == 'train':
            self.images, self.masks = self._load_train_valid_split(valid=False)
        elif split == 'valid':
            self.images, self.masks = self._load_train_valid_split(valid=True)
        elif split == 'test':
            self.images, self.masks = self._load_test_split()

    def __len__(self):
        return len(self.images)

    def _load_train_valid_split(self, valid=False) -> Tuple[list, list]:
        image_file = 'train_autoencoder.txt'
        mask_file = 'train_segmentation.txt'
        image_list = self._load_file_list(image_file)
        mask_list = self._load_file_list(mask_file)

        dataset_size = len(image_list)
        valid_size = int(self.valid_ratio * dataset_size)
        train_size = dataset_size - valid_size

        if valid:
            return image_list[train_size:], mask_list[train_size:]
        else:
            return image_list[:train_size], mask_list[:train_size]

    def _load_test_split(self) -> Tuple[list, list]:
        image_file = 'test_autoencoder.txt'
        mask_file = 'test_segmentation.txt'
        return self._load_file_list(image_file), self._load_file_list(mask_file)

    def _load_file_list(self, file_name: str) -> list:
        file_path = self.polyp_dir / file_name
        try:
            with open(file_path) as f:
                return sorted(line.strip() for line in f if line.strip().startswith('positive'))
        except FileNotFoundError:
            logging.error(f'File not found: {file_path}')
            return []
        except Exception as e:
            logging.error(f'Error reading file {file_path}: {e}')
            return []

    def __getitem__(self, idx):
        img_path = self.polyp_dir / self.images[idx]
        mask_path = self.polyp_dir / self.masks[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1:
            mask_np = mask_np / 255
        mask = Image.fromarray(mask_np, mode='L')

        if self.transforms:
            img, mask = self.transforms(img), self.transforms(mask)

        return img, mask

    @staticmethod
    def get_train_valid_and_test(polyp_dir, valid_ratio, transforms=None):
        train_set = PolypGen2021Dataset(polyp_dir, 'train', valid_ratio=valid_ratio, transforms=transforms[0])
        if valid_ratio > 0.0:
            valid_set = PolypGen2021Dataset(polyp_dir, 'valid', valid_ratio=valid_ratio, transforms=transforms[0])
        else:
            valid_set = None
        test_set = PolypGen2021Dataset(polyp_dir, 'test', valid_ratio=valid_ratio, transforms=transforms[1])
        return (train_set, valid_set, test_set)
