import logging
import os
import os.path
from typing import Union, Tuple, List

import numpy as np
import torch


def create_file_unsafe(filename: str):
    with open(filename, 'w'):
        pass


def create_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def create_file_parents(filename: str):
    dirname = os.path.dirname(filename)
    create_dirs(dirname)


def create_file(filename: str):
    if os.path.exists(filename):
        return

    try:
        create_file_unsafe(filename)
    except FileNotFoundError or OSError:
        create_file_parents(filename)
        create_file_unsafe(filename)


def file_prefix_name(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[0]

def file_suffix_name(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[1]

def save_model(filename, model, optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    checkpoint = dict()
    checkpoint["model"] = model.state_dict()
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    if lr_scheduler:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    for k, v in kwargs.items():
        checkpoint[k] = v

    try:
        torch.save(checkpoint, filename)
    except FileNotFoundError:
        create_file_parents(filename)
        torch.save(checkpoint, filename)


def load_model(filename: str, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        return checkpoint
    except FileNotFoundError as e:
        logging.error(f'File Not Found: {e}')
        raise e
    except Exception as e:
        logging.error(f'Error loading model: {e}')
        raise e


from pprint import pprint
from typing import TextIO
def print_model_info(model_src: str, output_stream: TextIO):
    checkpoint = load_model(model_src, torch.device("cpu"))
    pprint(checkpoint, stream=output_stream)

from torchinfo import summary
def summary_model_info(model_src: str, input_size: Tuple[int, int, int, int]):
    checkpoint = load_model(model_src, torch.device("cpu"))
    summary(checkpoint['model'], input_size=input_size)


def save_data(filename: str, data: Union[np.ndarray, torch.Tensor]) -> None:
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    try:
        np.save(filename, data)
    except FileNotFoundError as e:
        create_file_parents(filename)
        np.save(filename, data)


def load_data(filename: str) -> np.ndarray:
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

# torch shape: (B, C, H, W)
# numpy shape: (C, H, W)
# PIL.Image shape: (W, H, C)
