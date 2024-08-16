from matplotlib import pyplot as plt
import torch
import numpy as np
from typing import List, Union, Optional, Dict
from PIL import Image
from utils.metrics import metric_name_list

ImageType = Union[torch.Tensor, np.ndarray, Image.Image]


def save_image(filename: str):
  plt.savefig(filename)


def show_images(images: List[ImageType], 
                nrows: int, ncols: int, 
                scale: float=1.0, 
                titles: Optional[List[str]]=None):
  assert nrows > 0 and ncols > 0, 'nrows and ncols should be greater than 0'
  
  figsize = (ncols * scale, nrows * scale)
  _, axes = plt.subplots(nrows, ncols, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, images)):
    if torch.is_tensor(img):
      img = img.numpy()
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  plt.show()


def show_image_comparison(pre: ImageType, post: ImageType, mask: ImageType, 
                          titles: Optional[List[str]]=None, 
                          filename: Optional[str]=None):
  assert titles is None or len(titles) == 3, 'Images must have the same number of titles'
  
  figsize = (10, 10)
  _, axes = plt.subplots(2, 2, figsize=figsize)
  
  axes[0, 0].axis('off')
  axes[0, 1].axis('off')
  axes[1, 0].axis('off')
  axes[1, 1].axis('off')
  
  axes[0, 0].imshow(pre)
  axes[0, 1].imshow(post)
  axes[1, 0].imshow(mask)
  axes[1, 1].set_position([0.25, 0.1, 0.5, 0.5])
  
  if titles:
    axes[0, 0].set_title(titles[0])
    axes[0, 1].set_title(titles[1])
    axes[1, 0].set_title(titles[2])
  
  plt.subplots_adjust(wspace=0.1, hspace=0.3)
  save_image(filename) if filename else plt.show()


def draw_metrics(metrics: dict, colors: List[str], 
                 title: Optional[str] = None, 
                 selected: List[str]=metric_name_list, 
                 save_data: bool=True,
                 filename: Optional[str]=None):
  assert metrics is not None and len(selected) == len(colors), 'metrics must be specified for each metric type'
  
  plt.figure()
  
  metric_show = dict()
  for name in selected:
    metric_show[name] = metrics.get(name) 
  
  plt.ylim([0, 1])
  plt.bar(metric_show.keys(), height=metric_show.values(), color=colors)
  
  if title is not None:
    plt.title(title)
  
  save_image(filename) if filename else plt.show()


def draw_loss_graph(losses: List[float], 
                    title: Optional[str]=None, 
                    save_data: bool=True, 
                    filename: Optional[str]=None):
  assert losses is not None and len(losses) > 0, 'losses must be specified for each iteration'
  
  plt.figure()
  plt.plot([i+1 for i in range(len(losses))], losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  if title is not None:
    plt.title(title)
  
  save_image(filename) if filename else plt.show()
  