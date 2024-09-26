import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets.dataset_isic import *
from utils.datasets.dataset_drive import *
from utils.datasets.dataset_bowl import *
from utils.datasets.dataset_polyp import *
from torch.utils.data import DataLoader

from typing import Tuple

from utils import Transforms


def support_datasets():
    return 'ISIC2017 ISIC2018 ISIC2019 POLYPGEN2021 BOWL2018 DRIVE'.split()


def get_train_valid_and_test_loader(dataset_name, dataset_dir, batch_size,
                                    train_valid_test: Tuple[float, float, float],
                                    use_augment_enhance=True, resize=(512, 512), num_workers=0):
    boost_transform_group = Transforms.TransformBuilder().resize(resize).horizon_flip().vertical_flip().tensorize().build()
    default_transform_group = Transforms.TransformBuilder().resize(resize).tensorize().build()

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
            dataset_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
    else:
        valid_loader = None
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_train_and_test_loader(dataset_name, dataset_dir, batch_size, 
                              train_test_size: Tuple[float, float], 
                              use_augment_enhance=True, resize=(512, 512), num_workers=0):
    return get_train_valid_and_test_loader(
        dataset_name, dataset_dir, batch_size,
        (train_test_size[0], 0, train_test_size[1]), 
        use_augment_enhance=use_augment_enhance, 
        resize=resize, num_workers=num_workers)