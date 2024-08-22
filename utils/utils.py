import logging
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
import os
import wandb

def create_file(filepath):
  try:
    with open(filepath, 'r') as f:
      pass
  except FileNotFoundError:
    if not os.path.exists(os.path.dirname(filepath)):
      os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
      pass

def create_file_path_or_not(filepath):
  dirpath = os.path.dirname(filepath)
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)

def save_model(model, path):
  try:
    torch.save(model.state_dict(), path)
  except FileNotFoundError:
    create_file(path)
    torch.save(model.state_dict(), path)
    wandb.save(path)

def load_model(model, path):
  try:
    torch.load(model.state_dict(), path)
  except FileNotFoundError as e:
    logging.error(f'File Not Found: {e}')
    raise e

def save_data(filename: Union[str, Path], data: Union[np.ndarray, torch.Tensor]):
  if isinstance(data, torch.Tensor):
    data = data.cpu().detach().numpy()

  try:
    np.save(filename, data)
    wandb.save(filename)
  except FileNotFoundError as e:
    create_file(filename)
    np.save(filename, data)
    wandb.save(filename)


def load_data(filename: Union[str, Path]):
  try:
    data = np.load(filename)
    return data 
  except FileNotFoundError as e:
    logging.error(f'File Not Found: {e}')
    raise e

def tuple2list(t: Tuple):
  return list(t)

def list2tuple(l: List):
  return tuple(l)
