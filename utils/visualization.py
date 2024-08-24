from matplotlib import pyplot as plt
import torch
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from PIL import Image
from utils.utils import create_dirs, create_file_parents
import seaborn as sns

ImageType = Union[torch.Tensor, np.ndarray, Image.Image]
ArrayLike = Union[List[float], np.array, torch.Tensor]


def save_image(filename: str):
  create_file_parents(filename)
  plt.savefig(filename)
  plt.close()


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
  save_image(filename)


def save_metrics(metrics: dict, 
                 colors: List[str], 
                 filename: str,
                 selected: List[str],
                 title: Optional[str] = None):  
  assert len(colors) >= len(metrics)
  
  metrics_show = dict(filter(lambda item: item[0] in selected, metrics.items()))
  
  plt.figure()
  plt.ylim([0, 1])
  plt.bar(metrics_show.keys(), height=metrics_show.values(), color=colors[:len(selected)])
  
  if title:
    plt.title(title)
  
  save_image(filename)


def draw_xy_graph(values: List[float], gap: float, xlabel: str, ylabel: str, filename: str, title: Optional[str]=None):
  plt.figure()
  plt.plot([(i+1)*gap for i in range(len(values))], values)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  if title:
    plt.title(title)
  
  save_image(filename)


def draw_loss_graph(losses: ArrayLike, filename: str, title: Optional[str]=None):
  draw_xy_graph(losses, gap=1.0, xlabel="Epoch", ylabel="Loss", filename=filename, title=title)


def draw_heat_graph(possibility_matrix: np.array, filename: str, title: Optional[str]=None, x_ticks=False, y_ticks=False, x_label: str="", y_label: str=""):
  ax = sns.heatmap(possibility_matrix, xticklabels=x_ticks, yticklabels=y_ticks)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  if title:
    ax.set_title(title)

  save_image(filename)

def draw_attention_heat_graph(possibility_matrix: np.array, original_image: Union[np.array, Image.Image, torch.Tensor], filename: str, title: Optional[str]=None, x_ticks=False, y_ticks=False, x_label: str="", y_label: str=""):  
  """
    possiblity_matrix:
      shape: H, W
      value_range: [0, 1]
    original_image:
      shape: H, W, C
      value_range: No Limit
  """
  if isinstance(original_image, Image.Image):
    original_image = np.array(original_image)
  elif isinstance(original_image, torch.Tensor):
    original_image = original_image.detach().cpu().numpy()
    
  assert possibility_matrix.shape[:2] == original_image.shape[:2], "Both of them should be matched at the pixel level."
  
  plt.figure()
  plt.imshow(original_image)
  plt.imshow(possibility_matrix, cmap='YlOrRd', alpha=0.5)
  
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if not x_ticks:
    plt.xticks([])
  if not y_ticks:
    plt.yticks([])
  if title:
    plt.set_title(title)
  
  plt.tight_layout()
  save_image(filename)
