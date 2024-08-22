import logging
import torch
import numpy as np
from typing import Union, List, Any
import pandas as pd
from utils.utils import create_file
import os
import time

class CSVWriter:
  def __init__(self, filename: str):
    self.filename = filename
    self.df = pd.DataFrame()

  def get_headers(self):
    return self.df.columns.to_list()

  def get_data(self, header=None):
    try:
      return self.df.to_numpy() if header is None else self.df[header]
    except Exception as e:
      logging.error(f'No header: {header}')
      return None
      
  def write(self, header: str, data: Union[float, np.ndarray, torch.Tensor, List[Any]]):
    if isinstance(data, torch.Tensor):
      data = data.cpu().detach().numpy()
    
    try:
      self.df[header] = data
    except Exception as e:
      logging.warn(f"No header: {header}")
      self.write_headers([header])
      self.df[header] = data
    return self

  def write_headers(self, headers: List[str]):
    original_headers = self.df.columns.to_list()
    original_len = len(original_headers)
    
    headers = [header for header in original_headers if header not in headers] + headers
    for i in range(original_len, len(headers)):
      self.df[headers[i]] = None
    self.df = self.df.reindex(columns=headers)
    
    return self

  def flush(self):
    try:
      self.df.to_csv(self.filename, index=False, mode='w+', header=self.df.columns.tolist() if self.df.empty else False, encoding='utf-8')
    except (IOError, OSError) as e:
      logging.error(f"Error writing to CSV file: {e}")
    return self
