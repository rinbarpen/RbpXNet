import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets.dataset_isic import *
# from utils.datasets.dataset_drive import * 
from utils.datasets.dataset_polyp import PolypGen2021Dataset

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
  elif dataset_name == 'POLYPGEN':
    return PolypGen2021Dataset.get_train_valid_and_test(
      dataset_dir, valid_ratio=train_valid_test[1],
      transforms=select_transform_group)

def support_datasets():
  return 'ISIC2017 ISIC2018 ISIC2019 POLYPGEN2021 BOWL2018 DRIVE'.split()
