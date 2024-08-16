import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import *
from torchvision import transforms
import cv2
from random import random

SplitType = Literal['train', 'valid', 'test'] 
# class DriveDataset(Dataset):
#   # 'train' 'valid' 'test'
#   def __init__(self, dirpath, split: SplitType='train', transforms=None, tv_ratio=0.2):
#     self.dir = Path(dirpath)    
#     self.transforms = transforms

#     if split == 'train':
#       self.image_dir = self.dir / 'ISIC_2019_Training_Input/'
#       self.gt_file = self.dir / 'ISIC_2019_Training_Input/ISIC_2019_Training_GroundTruth.csv'
#     elif split == 'valid':
#       self.image_dir = self.dir / 'ISIC_2019_Training_Input/'
#       self.gt_file = self.dir / 'ISIC_2019_Training_Input/ISIC_2019_Training_GroundTruth.csv'
#     elif split == 'test':
#       self.image_dir = self.dir / 'ISIC_2019_Test_Input/'
#       self.gt_file = self.dir / 'ISIC_2019_Test_Metadata.csv'
    

#   def __len__(self):
#     return len(self.images)
  
#   def __getitem__(self, idx):
#     image = self.images[idx]
    
#     image_path = self.image_dir / image
#     image = cv2.imread(image_path).resize((512, 512))
    
#     ground_truthes = self.gt_data
    
#     return image, mask
