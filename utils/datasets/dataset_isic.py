from torch.utils.data import Dataset
from typing import Literal, List, Tuple
import os
from pathlib import Path
import pandas as pd

from torchvision import transforms
from PIL import Image
import sys



class ISIC2017Dataset(Dataset):
  def __init__(self, isic_dir, split: Literal['train', 'test', 'valid'], transformers=None):
    super(ISIC2017Dataset, self).__init__()
    self.isic_dir = Path(isic_dir)
    self.transformers = transformers

    if split == 'train': 
      self.images_dir = self.isic_dir / 'ISIC-2017_Training_Data'
      self.masks_dir = self.isic_dir / 'ISIC-2017_Training_Part1_GroundTruth'
    elif split == 'test':
      self.images_dir = self.isic_dir / 'ISIC-2017_Test_v2_Data'
      self.masks_dir = self.isic_dir / 'ISIC-2017_Test_v2_Part1_GroundTruth'
    elif split == 'valid':
      self.images_dir = self.isic_dir / 'ISIC-2017_Validation_Data'
      self.masks_dir = self.isic_dir / 'ISIC-2017_Validation_Part1_GroundTruth'

    if not self.images_dir.exists() or not self.masks_dir.exists():
      raise FileNotFoundError(f"Directory not found: {self.images_dir} or {self.masks_dir}")


    self.images = sorted([f for f in self.images_dir.glob('*.jpg')])
    self.masks = sorted([f for f in self.masks_dir.glob('*.png')])

    if len(self.images) != len(self.masks):
      raise ValueError("Number of images and masks do not match.")

  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx):
    try:
      img_path, mask_path = self.images[idx], self.masks[idx]

      img = Image.open(img_path).convert('RGB')
      mask = Image.open(mask_path).convert('L')
    except Exception as e:
      raise RuntimeError(f"Error loading image or mask at index {idx}: {e}")

    if self.transformers:
      img, mask = self.transformers(img, mask)

    return img, mask

  def get_images(self):
    return self.images
  
  def get_masks(self):
    return self.masks

  @staticmethod
  def get_train_valid_and_test(isic_dir, transformers=None):
    train_set = ISIC2017Dataset(isic_dir, 'train', transformers=transformers)
    valid_set = ISIC2017Dataset(isic_dir, 'valid', transformers=transformers)
    test_set  = ISIC2017Dataset(isic_dir, 'test',  transformers=transformers)
    return (train_set, valid_set, test_set)


class ISIC2018Dataset(Dataset):
  def __init__(self, isic_dir, split: Literal['train', 'test', 'valid'], transformers=None):
    super(ISIC2018Dataset, self).__init__()
    self.isic_dir = Path(isic_dir)
    self.transformers = transformers

    if split == 'train': 
      self.images_dir = self.isic_dir / 'ISIC2018_Task1-2_Training_Input'
      self.masks_dir = self.isic_dir / 'ISIC2018_Task1_Training_GroundTruth'
    elif split == 'test':
      self.images_dir = self.isic_dir / 'ISIC2018_Task1-2_Test_Input'
      self.masks_dir = self.isic_dir / 'ISIC2018_Task1_Test_GroundTruth'
    elif split == 'valid':
      self.images_dir = self.isic_dir / 'ISIC2018_Task1-2_Validation_Input'
      self.masks_dir = self.isic_dir / 'ISIC2018_Task1_Validation_GroundTruth'

    if not self.images_dir.exists() or not self.masks_dir.exists():
      raise FileNotFoundError(f"Directory not found: {self.images_dir} or {self.masks_dir}")


    self.images = sorted([f for f in self.images_dir.glob('*.jpg')])
    self.masks = sorted([f for f in self.masks_dir.glob('*.png')])
  
    if len(self.images) != len(self.masks):
      raise ValueError("Number of images and masks do not match.")


  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx):
    try:
      img_path, mask_path = self.images[idx], self.masks[idx]

      img = Image.open(img_path).convert('RGB')
      mask = Image.open(mask_path).convert('L')
    except Exception as e:
      raise RuntimeError(f"Error loading image or mask at index {idx}: {e}")

    if self.transformers:
      img, mask = self.transformers(img, mask)

    return img, mask

  def get_images(self):
    return self.images
  
  def get_masks(self):
    return self.masks

  @staticmethod
  def get_train_valid_and_test(isic_dir, transformers=None):
    train_set = ISIC2018Dataset(isic_dir, 'train', transformers=transformers)
    valid_set = ISIC2018Dataset(isic_dir, 'valid', transformers=transformers)
    test_set  = ISIC2018Dataset(isic_dir, 'test', transformers=transformers)
    return (train_set, valid_set, test_set)


class ISIC2019Dataset(Dataset):
  def __init__(self, isic_dir, split: Literal['train', 'test', 'valid'], train_valid_test: List[float], transformers=None):
    super(ISIC2019Dataset, self).__init__()
    self.isic_dir = Path(isic_dir)
    self.transformers = transformers
    
    self.images_dir = self.isic_dir / 'ISIC_2019_Training_Input'

    self.images = self._get_images(split, train_valid_test)
    self.label_df = pd.read_csv(self.images_dir / 'ISIC_2019_Training_GroundTruth')


  def _get_images(self, split: Literal['train', 'test', 'valid'], train_valid_test: List[float]):
    full_images = sorted([f for f in self.images_dir.glob('*.jpg')])
    n = len(full_images)
    if split == 'train':
      right = int(n * train_valid_test[0])
      return full_images[:right]
    elif split == 'valid':
      left = int(n * train_valid_test[0])
      right = int(n * (train_valid_test[0]+train_valid_test[1]))
      return full_images[left:right]
    elif split == 'test':
      left = int(n * train_valid_test[1])
      return full_images[left:]


  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image_path = self.images[idx]
    label = self.label_df[self.label_df['image'] == image_path].values[1:]

    image = Image.open(image_path).convert('RGB')

    if self.transformers:
      image = self.transformers(image)

    return image, label

  @staticmethod
  def get_train_valid_and_test(isic_dir, train_valid_test:List[float], transformers=None):
    train_set = ISIC2019Dataset(isic_dir, 'train', train_valid_test, transformers=transformers)
    valid_set = ISIC2019Dataset(isic_dir, 'valid', train_valid_test, transformers=transformers)
    test_set  = ISIC2019Dataset(isic_dir, 'test', train_valid_test, transformers=transformers)
    return (train_set, valid_set, test_set)
