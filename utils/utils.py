import torch
import numpy as np
from pathlib import Path
from typing import Union

def save_model(model, path):
  torch.save(model.state_dict(), path)

def load_model(model, path):
  torch.load(model.state_dict(), path)

def save_data(filename: Union[str, Path], data: Union[np.ndarray, torch.Tensor]):
  if isinstance(data, torch.Tensor):
    data = data.cpu().detach().numpy()

  np.save(filename, data)

def load_data(filename: Union[str, Path]):
  data = np.load(filename)
  return data 

def tuple2list(t):
  return list(t)

def list2tuple(l):
  return tuple(l)
