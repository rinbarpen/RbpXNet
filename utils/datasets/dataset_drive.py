import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import Literal

from utils.Transforms import TransformBuilder


SplitType = Literal['train', 'valid', 'test'] 

class DriveDataset(Dataset):
  # 'train' 'valid' 'test'
  def __init__(self, dirpath, split: SplitType='train', tv_ratio=0.2, transforms=None):
    self.dir = Path(dirpath)
    self.transforms = transforms

    if split == 'train':
      self.image_dir = self.dir / 'training' / 'images'
      self.mask_dir = self.dir / 'training' / '1st_manual'
    elif split == 'valid':
      self.image_dir = self.dir / 'training' / 'images'
      self.mask_dir = self.dir / 'training' / '1st_manual'
    elif split == 'test':
      self.image_dir = self.dir / 'test' / 'images'
      self.mask_dir = self.dir / 'test' / '1st_manual'

    self.images = [self.image_dir / image for image in self.image_dir.glob('*tif')]
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
    
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    if self.transforms:
      image, mask = self.transforms[0](image), self.transforms[1](mask)
    
    return image, mask, os.path.splitext(os.path.basename(mask_path))[0], mask.shape
  
  @staticmethod
  def collate_fn(batch):
    images = torch.stack([item[0] for item in batch]) 
    masks = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    original_sizes = [item[3] for item in batch]
    return images, masks, filenames, original_sizes
  
  @staticmethod
  def get_train_valid_and_test(drive_dir, tv_ratio=0.2, transforms=None):
    train_set = DriveDataset(drive_dir, 'train', tv_ratio, transforms=transforms[0])
    valid_set = DriveDataset(drive_dir, 'valid', tv_ratio, transforms=transforms[0])
    test_set  = DriveDataset(drive_dir, 'test',  tv_ratio, transforms=transforms[1])
    return (train_set, valid_set, test_set)
