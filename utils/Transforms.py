from typing import Tuple, Union, TypedDict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class BoostTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]):
        return self.transform(image)

class NormDict(TypedDict):
    mean: list[float]
    std: list[float]

class TransformBuilder:
    def __init__(self):
        self._to_PILImage = False
        self._resize = None
        self._rotation = 0.0
        self._horizon_flip = False
        self._vertical_flip = False
        self._tensorize = False
        self._norm = NormDict()

    def resize(self, resize: Tuple[int, int]):
        self._resize = resize
        return self

    def rotation(self, rotation: float):
        self._rotation = rotation
        return self

    def horizon_flip(self):
        self._horizon_flip = True
        return self

    def vertical_flip(self):
        self._vertical_flip = True
        return self

    def tensorize(self):
        self._tensorize = True
        return self 

    def normalize(self, mean, std):
        self._norm["mean"] = mean
        self._norm["std"] = std
        return self

    def norm_rgb(self):
        self._norm["mean"] = [0.485, 0.456, 0.406]
        self._norm["std"] = [0.229, 0.224, 0.225]
        return self

    def norm_gray(self):
        self._norm["mean"] = [0.5]
        self._norm["std"] = [0.5]
        return self

    def to_PILImage(self):
        self._to_PILImage = True
        return self

    def build(self):
        transform = []
        if self._to_PILImage:
            transform.append(transforms.ToPILImage())

        if self._resize:
            transform.append(transforms.Resize(self._resize))

        if self._horizon_flip:
            transform.append(transforms.RandomHorizontalFlip())
        if self._vertical_flip:
            transform.append(transforms.RandomVerticalFlip())

        if self._rotation != 0.0:
            transform.append(transforms.RandomRotation(self._rotation))

        if self._tensorize:
            transform.append(
                transforms.ToTensor(),
            )

        if self._norm:
            transform.append(
                transforms.Normalize(mean=self._norm["mean"], std=self._norm["std"])
            )

        return BoostTransform(transforms.Compose(transform))


def get_rgb_image_transform(resize: Tuple[int, int], rotation=0.0, flip=True, norm=True):
    trans = TransformBuilder().resize(resize).rotation(rotation)
    if flip:
        trans = trans.horizon_flip().vertical_flip()
    trans = trans.tensorize()
    if norm:
        trans = trans.norm_rgb()
    return trans.build()

def get_mask_transform(resize: Tuple[int, int], rotation=0.0, flip=True, norm=True):
    trans = TransformBuilder().resize(resize).rotation(rotation)
    if flip:
        trans = trans.horizon_flip().vertical_flip()
    trans = trans.tensorize()
    if norm:
        trans = trans.norm_gray()
    return trans.build()
