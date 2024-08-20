import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
import unittest
from utils.datasets.dataset import support_datasets
from utils.datasets.dataset_polyp import *
from utils.datasets.dataset_bowl import *
from utils.datasets.dataset_isic import *
from utils.datasets.dataset_drive import *
from utils.Transforms import TransformBuilder, get_rgb_image_transform, get_mask_transform


class TestDataset(unittest.TestCase):
  def setUp(self):
    self.datasets = support_datasets()
    self.dir_prefix = 'I:/AI/Data/'
    self.data_dirs = {ds: self.dir_prefix + ds for ds in self.datasets}
    self.transforms = (get_rgb_image_transform((512, 512)),
                       get_mask_transform((512, 512)))
    self.batch_size = 1

    print(f'{repr(self.data_dirs)}')

  def tearDown(self):
    pass

  def test_isic2017_dataset_get(self, *args, **kwargs):
    self.isic2017_dir = self.data_dirs['ISIC2017']
    print(f'{self.isic2017_dir=}')
    isic2017_dataset = ISIC2017Dataset(self.isic2017_dir, 'train', self.transforms)
    loader = DataLoader(isic2017_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'Dataset: {len(isic2017_dataset)=}')
    print(f'DataLoader: {len(loader)=}')

    # peek
    for image, label in loader:
      print(f'Image shape: {image.shape}')
      print(f'Label shape: {label.shape}')
      break  

  def test_isic2018_dataset_get(self, *args, **kwargs):
    self.isic2018_dir = self.data_dirs['ISIC2018']
    print(f'{self.isic2018_dir=}')
    isic2018_dataset = ISIC2018Dataset(self.isic2018_dir, 'train', self.transforms)
    loader = DataLoader(isic2018_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'Dataset: {len(isic2018_dataset)=}')
    print(f'DataLoader: {len(loader)=}')

    # peek
    for image, label in loader:
      print(f'Image shape: {image.shape}')
      print(f'Label shape: {label.shape}')
      break  

  def test_polyp2021_dataset_get(self, *args, **kwargs):
    self.polyp_dir = self.data_dirs['POLYPGEN2021']
    print(f'{self.polyp_dir=}')
    polyp_dataset = PolypGen2021Dataset(self.polyp_dir, 'train', 0.2, self.transforms)
    loader = DataLoader(polyp_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'Dataset: {len(polyp_dataset)=}')
    print(f'DataLoader: {len(loader)=}')
  
  def test_bowl2018_dataset_get(self, *args, **kwargs):
    self.bowl_dir = self.data_dirs['BOWL2018']
    print(f'{self.bowl_dir=}')
    bowl_dataset = Bowl2018Dataset(self.bowl_dir, 'train', [0.7, 0.2, 0.1], self.transforms)
    loader = DataLoader(bowl_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'Dataset: {len(bowl_dataset)=}')
    print(f'DataLoader: {len(loader)=}') 

    # peek
    for image, label in loader:
      print(f'Image shape: {image.shape}')
      print(f'Label shape: {label[0].shape}')
      break 

  def test_drive_dataset_get(self, *args, **kwargs):
    self.drive_dir = self.data_dirs['DRIVE']
    print(f'{self.drive_dir=}')
    drive_dataset = DriveDataset(self.drive_dir, 'train', 0.2, self.transforms)
    loader = DataLoader(drive_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'Dataset: {len(drive_dataset)=}')
    print(f'DataLoader: {len(loader)=}')

    # peek
    for image, label in loader:
      print(f'Image shape: {image.shape}')
      print(f'Label shape: {label.shape}')
      break  
  
