import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import Literal


SplitType = Literal['train', 'valid', 'test'] 

class DriveDataset(Dataset):
  # 'train' 'valid' 'test'
  def __init__(self, dirpath, split: SplitType='train', tv_ratio=0.2, transforms=None):
    self.dir = Path(dirpath)
    self.transforms = transforms

    if split == 'train':
      self.image_dir = self.dir / 'training' / 'images'
      self.mask_dir = self.dir / 'training' / 'mask'
    elif split == 'valid':
      self.image_dir = self.dir / 'training' / 'images'
      self.mask_dir = self.dir / 'training' / 'mask'
    elif split == 'test':
      self.image_dir = self.dir / 'test' / 'images'
      self.mask_dir = self.dir / 'test' / 'mask'

    self.images = [self.image_dir / image for image in os.listdir(self.image_dir) if image.endswith('.tif')]
    self.masks = [self.mask_dir / mask for mask in os.listdir(self.mask_dir) if mask.endswith('.gif')]
    
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
    
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    if self.transforms:
      image, mask = self.transforms[0](image), self.transforms[1](mask)
    
    return image, mask

  @staticmethod
  def get_train_valid_and_test(drive_dir, tv_ratio=0.2, transforms=None):
    train_set = DriveDataset(drive_dir, 'train', tv_ratio, transforms=transforms)
    valid_set = DriveDataset(drive_dir, 'valid', tv_ratio, transforms=transforms)
    test_set  = DriveDataset(drive_dir, 'test',  tv_ratio, transforms=transforms)
    return (train_set, valid_set, test_set)
