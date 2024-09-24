import logging
from typing import Union, List, Any, Dict

import numpy as np
import pandas as pd
import torch


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

    def write(self, header: str, data: Union[np.ndarray, torch.Tensor, List[Any]]):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        self.df[header] = list(data)
        return self

    def writes(self, datium: Dict[str, Union[np.ndarray, torch.Tensor, List[Any]]]):
        for header, data in datium.items():
            if isinstance(data, torch.Tensor):
                writen_data = data.cpu().detach().numpy()
            else:
                writen_data = data
            self.df[header] = [writen_data]

        return self

    def write_headers(self, headers: List[str]):
        original_headers = self.df.columns.to_list()
        original_len = len(original_headers)

        headers = [header for header in headers if header not in original_headers] + original_headers
        for i in range(original_len, len(headers)):
            self.df[headers[i]] = None
        self.df = self.df.reindex(columns=headers)

        return self

    def flush(self):
        try:
            self.df.to_csv(self.filename, index=False, mode='w+',
                           header=self.df.columns.tolist(), encoding='utf-8')
        except (IOError, OSError) as e:
            logging.error(f"Error writing to CSV file: {e}")
        return self
