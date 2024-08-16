import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.datasets.dataset_isic import *
# from utils.datasets.dataset_drive import * 

from typing import Literal, List, Union

from torchvision import transforms
from utils.datasets.dataset_polyp import PolypGen2021Dataset


class CustomTransform:
  def __init__(self, resize: Tuple[int, int], rotation: float):
    self.image_transform = transforms.Compose([
      transforms.Resize(resize),      # 调整图像大小
      transforms.RandomHorizontalFlip(),   # 随机水平翻转
      transforms.RandomVerticalFlip(),     # 随机垂直翻转
      transforms.RandomRotation(rotation), # 随机旋转
      transforms.ToTensor(),               # 转换为 Tensor
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    self.mask_transform = transforms.Compose([
      transforms.Resize(resize),      # 调整掩膜大小
      transforms.ToTensor()               # 转换为 Tensor
    ])

  def __call__(self, image: Image.Image, mask: Image.Image):
    image = self.image_transform(image)
    mask = self.mask_transform(mask)
    return image, mask


class HybridDataset(Dataset):
  def __init__(self, datasets, data_dirs, split, transformers=None):
    self.datasets = datasets
    self.transformers = transformers

    self.images, self.masks = [], []
    for dataset, data_dir in zip(datasets, data_dirs):
      if dataset == 'ISIC2017':
        x = ISIC2017Dataset(data_dir, split, transformers=transformers)
        self.images.extend(x.get_images())
        self.masks.extend(x.get_masks())
      elif dataset == 'ISIC2018':
        x = ISIC2018Dataset(data_dir, split, transformers=transformers)
        self.images.extend(x.get_images())
        self.masks.extend(x.get_masks())

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path, mask_path = self.images[idx], self.masks[idx]
    
    img = Image.open(img_path).convert('RGB') 
    mask = Image.open(mask_path).convert('L')

    if self.transformers:
      img, mask = self.transformers(img, mask)

    return img, mask 


def get_dataset(datasets: Union[str, List[str]], 
                data_dirs: Union[str, List[str]], 
                split: Literal['train', 'valid', 'test'], 
                train_valid_test: List[float],
                resize: Tuple[int, int]=(512, 512)):
  custom_trans = CustomTransform(resize, 0)
  default_trans = transforms.Compose([
    transforms.Resize(resize),      # 调整图像大小
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.RandomVerticalFlip(),     # 随机垂直翻转
    transforms.RandomRotation(0), # 随机旋转
    transforms.ToTensor(),               # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
  ])


  if isinstance(datasets, list):
    return HybridDataset(datasets, data_dirs, split, transformers=custom_trans)

  dataset = datasets
  data_dir = data_dirs

  if dataset == 'ISIC2017':
    return ISIC2017Dataset(data_dir, split, transformers=custom_trans)
  elif dataset == 'ISIC2018':
    return ISIC2018Dataset(data_dir, split, transformers=custom_trans)
  elif dataset == 'ISIC2019':
    return ISIC2019Dataset(data_dir, split, train_valid_test, transformers=default_trans)
  elif dataset == 'POLYPGEN':
    return PolypGen2021Dataset(data_dir, split, valid_ratio=train_valid_test[1], transformers=custom_trans)
  else:
    raise ValueError(f'Unsupported dataset: {dataset}')

def support_datasets():
  return 'ISIC2017 ISIC2018'.split()
