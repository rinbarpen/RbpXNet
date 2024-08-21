import torch
import numpy as np
from typing import Union, List, Any
import pandas as pd

class CSVWriter:
  def __init__(self, filename: str):
    self.filename = filename
    self.df = pd.read_csv(self.filename)

  def get_headers(self):
    try:
      return self.df[0]
    except (IOError, OSError) as e:
      print(f"Error reading CSV file: {e}")
    return []

  def write(self, header: str, data: Union[np.ndarray, torch.Tensor, List[Any]]):
    try:
      if header not in self.df.columns:
        self.df[header] = pd.Series(data)
      else:
        self.df.loc[:, header] = data
      self.df.to_csv(self.filename, index=False, mode='a', header=self.df.columns.tolist() if self.df.empty else False)
    except (IOError, OSError) as e:
      print(f"Error writing to CSV file: {e}")
    return self

  def write_headers(self, headers: List[str]):
    try:
      self.df[0] = headers
    except (IOError, OSError) as e:
      print(f"Error writing header to CSV file: {e}")
    return self

