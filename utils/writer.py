import torch
import csv
import numpy as np
from typing import Union, List, Any
from abc import *
from functools import overloaded

class DataWriter(ABC):
  def __init__(self, filename):
    self.filename = filename
  
  @abstractmethod
  def write(self, datium: Union[np.ndarray, torch.Tensor, List[Any]]):
    pass

class CSVWriter(DataWriter):
  def __init__(self, filename):
    super().__init__(filename)

  @overloaded  
  def write(self, datium: Union[np.ndarray, torch.Tensor, List[Any]]):
    with open(self.filename, 'a', newline='\n') as f:
      writer = csv.writer(f)
      if isinstance(datium, (np.ndarray, torch.Tensor)):
        datium = datium.tolist()
      if isinstance(datium, list):
        for data in datium:
          writer.writerow(data)
