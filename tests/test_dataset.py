import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
import unittest
from utils.datasets.dataset import get_dataset, support_datasets, CustomTransform, HybridDataset


class TestDataset(unittest.TestCase):
  def setUp(self):
    self.datasets = support_datasets()
    self.dir_prefix = 'I:/AI/Data/'
    self.data_dirs = [self.dir_prefix + ds for ds in self.datasets]
    self.transforms = CustomTransform(resize=(512, 512), rotation=0)
    self.batch_size = 1

  def tearDown(self):
    pass

  def test_get_dataset(self):
    for dataset, data_dir in zip(self.datasets, self.data_dirs):
      # train dataset test
      ds = get_dataset(dataset, data_dir, 'train', self.transforms)
      loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
      print(f'{dataset}[Train] Dataset: {len(ds)=}')
      print(f'{dataset}[Train] DataLoader: {len(loader)=}')
      # valid dataset test
      ds = get_dataset(dataset, data_dir, 'valid', self.transforms)
      loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
      print(f'{dataset}[Validate] Dataset: {len(ds)=}')
      print(f'{dataset}[Validate] DataLoader: {len(loader)=}')
      # test dataset test
      ds = get_dataset(dataset, data_dir, 'test', self.transforms)
      loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
      print(f'{dataset}[Test] Dataset: {len(ds)=}')
      print(f'{dataset}[Test] DataLoader: {len(loader)=}')

  def test_get_hybrid_dataset(self):
    # train dataset test
    ds = get_dataset(self.datasets, self.data_dirs, 'train', self.transforms) # type: ignore
    loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f'{self.datasets}[Train] Dataset: {len(ds)=}')
    print(f'{self.datasets}[Train] DataLoader: {len(loader)=}')
    # valid dataset test
    ds = get_dataset(self.datasets, self.data_dirs, 'valid', self.transforms) # type: ignore
    loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f'{self.datasets}[Validate] Dataset: {len(ds)=}')
    print(f'{self.datasets}[Validate] DataLoader: {len(loader)=}')
    # test dataset test
    ds = get_dataset(self.datasets, self.data_dirs, 'test', self.transforms) # type: ignore
    loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f'{self.datasets}[Test] Dataset: {len(ds)=}')
    print(f'{self.datasets}[Test] DataLoader: {len(loader)=}')

