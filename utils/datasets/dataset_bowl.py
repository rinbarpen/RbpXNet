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
    def get_train_valid_and_test(bowl_dir, train_valid_test: Tuple[float, float, float], transforms=None):
        train_set = Bowl2018Dataset(bowl_dir, 'train', train_valid_test, transforms=transforms[0])
        if train_valid_test[1] > 0.0:
            valid_set = Bowl2018Dataset(bowl_dir, 'valid', train_valid_test, transforms=transforms[0])
        else:
            valid_set = None
        test_set  = Bowl2018Dataset(bowl_dir, 'test', train_valid_test, transforms=transforms[1])
        return (train_set, valid_set, test_set)
