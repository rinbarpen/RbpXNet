from torchvision import transforms
from typing import Tuple
from PIL import Image


class BoostTransform:
  def __init__(self, transform):
    self.transform = transforms.Compose(transform)

  def __call__(self, image: Image.Image):
    return self.transform(image)


class TransformBuilder:
  def __init__(self, resize: Tuple[int, int]):
    self._resize = resize
    self._rotation = 0.0
    self._horizon_flip = False 
    self._vertical_flip = False
    self._tensorize = False
    self._norm = False

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
  
  def normalize(self):
    self._norm = True
    return self

  def build(self):
    transform = [
      transforms.Resize(self._resize)
    ]

    if self._horizon_flip:
      transform.append(
        transforms.RandomHorizontalFlip()) 
    if self._vertical_flip:
      transform.append(
        transforms.RandomVerticalFlip()) 
      
    if self._rotation != 0.0:
      transform.append(
        transforms.RandomRotation(self._rotation)) 

    if self._tensorize:
      transform.append(
        transforms.ToTensor(),                
      )

    if self._norm:
      transform.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      )

    return BoostTransform(transform)


def get_rgb_image_transform(resize, rotation=0.0, flip=True):
  if flip: 
    return TransformBuilder(resize).rotation(rotation).horizon_flip().vertical_flip().tensorize().normalize().build()
  else: 
    return TransformBuilder(resize).rotation(rotation).tensorize().normalize().build()
  
def get_mask_transform(resize, rotation=0.0, flip=True):
  if flip: 
    return TransformBuilder(resize).rotation(rotation).horizon_flip().vertical_flip().tensorize().build()
  else: 
    return TransformBuilder(resize).rotation(rotation).tensorize().build()

