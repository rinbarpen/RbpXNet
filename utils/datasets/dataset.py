import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets.dataset_isic import *
from utils.datasets.dataset_drive import * 
from utils.datasets.dataset_bowl import * 
from utils.datasets.dataset_polyp import *
from torch.utils.data import DataLoader

from typing import Literal, List, Union

from utils import Transforms


def get_train_valid_and_test(dataset_name, dataset_dir, train_valid_test: List[float], use_augment_enhance=True):
  resize = (512, 512)
  boost_transform_group = (
    Transforms.TransformBuilder(resize)
    .rotation(0.0)
    .horizon_flip()
    .vertical_flip()
    .tensorize()
    .normalize()
    .build(), 
    Transforms.TransformBuilder(resize)
    .rotation(0.0)
    .horizon_flip()
    .vertical_flip()
    .tensorize()
    .build())
  default_transform_group = (
    Transforms.TransformBuilder(resize)
    .tensorize()
    .build(), 
    Transforms.TransformBuilder(resize)
    .tensorize()
    .build())

  select_transform_group = boost_transform_group if use_augment_enhance else default_transform_group

  if dataset_name == 'ISIC2017':
    return ISIC2017Dataset.get_train_valid_and_test(
      dataset_dir, 
      transforms=select_transform_group)
  elif dataset_name == 'ISIC2018':
    return ISIC2018Dataset.get_train_valid_and_test(
      dataset_dir, 
      transforms=select_transform_group)
  elif dataset_name == 'ISIC2019':
    return ISIC2019Dataset.get_train_valid_and_test(
      dataset_dir, train_valid_test, 
      transforms=select_transform_group)
  elif dataset_name == 'POLYPGEN2021':
    return PolypGen2021Dataset.get_train_valid_and_test(
      dataset_dir, valid_ratio=train_valid_test[1],
      transforms=select_transform_group)
  elif dataset_name == 'BOWL2018':
    return Bowl2018Dataset.get_train_valid_and_test(
      dataset_dir, train_valid_test,
      transforms=select_transform_group)
  elif dataset_name == 'DRIVE':
    return DriveDataset.get_train_valid_and_test(
      dataset_dir, train_valid_test[1],
      transforms=select_transform_group)

def support_datasets():
  return 'ISIC2017 ISIC2018 ISIC2019 POLYPGEN2021 BOWL2018 DRIVE'.split()


def get_train_valid_and_test_loader(dataset_name, dataset_dir, batch_size, train_valid_test: List[float], use_augment_enhance=True, resize=(512, 512), num_workers=0):
  boost_transform_group = (
    Transforms.TransformBuilder(resize)
    .rotation(0.0)
    .horizon_flip()
    .vertical_flip()
    .tensorize()
    .normalize()
    .build(), 
    Transforms.TransformBuilder(resize)
    .rotation(0.0)
    .horizon_flip()
    .vertical_flip()
    .tensorize()
    .build())
  default_transform_group = (
    Transforms.TransformBuilder(resize)
    .tensorize()
    .build(), 
    Transforms.TransformBuilder(resize)
    .tensorize()
    .build())

  select_transform_group = (boost_transform_group if use_augment_enhance else default_transform_group, default_transform_group)

  if dataset_name == 'ISIC2017':
    train_dataset, valid_dataset, test_dataset = ISIC2017Dataset.get_train_valid_and_test(
      dataset_dir, 
      transforms=select_transform_group)
  elif dataset_name == 'ISIC2018':
    train_dataset, valid_dataset, test_dataset = ISIC2018Dataset.get_train_valid_and_test(
      dataset_dir, 
      transforms=select_transform_group)
  elif dataset_name == 'ISIC2019':
    train_dataset, valid_dataset, test_dataset = ISIC2019Dataset.get_train_valid_and_test(
      dataset_dir, train_valid_test, 
      transforms=select_transform_group)
  elif dataset_name == 'POLYPGEN2021':
    train_dataset, valid_dataset, test_dataset = PolypGen2021Dataset.get_train_valid_and_test(
      dataset_dir, valid_ratio=train_valid_test[1],
      transforms=select_transform_group)
  elif dataset_name == 'BOWL2018':
    train_dataset, valid_dataset, test_dataset = Bowl2018Dataset.get_train_valid_and_test(
      dataset_dir, train_valid_test,
      transforms=select_transform_group)
  elif dataset_name == 'DRIVE':
    train_dataset, valid_dataset, test_dataset = DriveDataset.get_train_valid_and_test(
      dataset_dir, train_valid_test[1],
      transforms=select_transform_group)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, 
                            collate_fn=DriveDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=DriveDataset.collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=DriveDataset.collate_fn)
    return train_loader, valid_loader, test_loader

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
  test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
  return train_loader, valid_loader, test_loader
