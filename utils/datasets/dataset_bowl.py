import os
import os.path
from pathlib import Path
from typing import Literal, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Bowl2018Dataset(Dataset):
    def __init__(self, bowl_dir, split: Literal['train', 'valid', 'test'], train_valid_test: Tuple[float, float, float], transforms=None):
        super(Bowl2018Dataset, self).__init__()
        self.bowl_dir = Path(bowl_dir)
        self.transforms = transforms

        self.images_dir = self.bowl_dir / 'stage1_train'
        if split == 'train':
            self.images, self.masks = self._load_image_and_mask(split, train_valid_test)
        elif split == 'valid':
            self.images, self.masks = self._load_image_and_mask(split, train_valid_test)
        elif split == 'test':
            self.images, self.masks = self._load_image_and_mask(split, train_valid_test)

    def __len__(self):
        return len(self.images)

    def _load_image_and_mask(self, split, train_valid_test: Tuple[float, float, float]) -> Tuple[list, list]:
        images, masks = [], [[]]
        for d in os.listdir(self.images_dir):
            images.extend([x for x in (self.images_dir / d / 'images').glob('.png')])
            masks.append([x for x in (self.images_dir / d / 'masks').glob('.png')])

        if split == 'train':
            right = int(len(images) * train_valid_test[0])
            images = images[:right]
            masks = masks[:right]
        elif split == 'valid':
            left = int(len(images) * train_valid_test[0])
            right = int(len(images) * train_valid_test[1])
            images = images[left:right]
            masks = masks[left:right]
        elif split == 'test':
            left = int(len(images) * train_valid_test[1])
            images = images[left:]
            masks = masks[left:]

        return images, masks

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_paths = self.masks[idx]

        img = Image.open(image_path).convert('RGB')
        masks = []
        
        # TODO: check if image is already present
        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask, dtype=np.float32)
            if mask_np.max() > 1:
                mask_np = mask_np / 255
            mask = Image.fromarray(mask_np, mode='L')
            masks.append(mask)

        if self.transforms:
            img = self.transforms(img)
            masks = [self.transforms(mask) for mask in masks]

        return img, masks

    @staticmethod
    def get_train_valid_and_test(bowl_dir, train_valid_test: Tuple[float, float, float], transforms=None):
        train_set = Bowl2018Dataset(bowl_dir, 'train', train_valid_test, transforms=transforms[0])
        if train_valid_test[1] > 0.0:
            valid_set = Bowl2018Dataset(bowl_dir, 'valid', train_valid_test, transforms=transforms[0])
        else:
            valid_set = None
        test_set  = Bowl2018Dataset(bowl_dir, 'test', train_valid_test, transforms=transforms[1])
        return (train_set, valid_set, test_set)
