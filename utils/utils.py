import logging
import os
import os.path
from pathlib import Path

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


def file_prefix_name(filepath: str|Path):
    if isinstance(filepath, str):
        return os.path.splitext(os.path.basename(filepath))[0]
    elif isinstance(filepath, Path):
        return filepath.stem

def file_suffix_name(filepath: str|Path):
    if isinstance(filepath, str):
        return os.path.splitext(os.path.basename(filepath))[1]
    elif isinstance(filepath, Path):
        return filepath.suffix

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
    except Exception:
        create_file_parents(filename)
        torch.save(checkpoint, filename)


def load_model(filename: str, device: torch.device|str) -> dict:
    try:
        if isinstance(device, str):
            device = torch.device(device)
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
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
    checkpoint = load_model(model_src, "cpu")
    pprint(checkpoint, stream=output_stream)

from torchinfo import summary
def summary_model_info(model_src: str|torch.nn.Module, input_size: tuple[int, int, int, int]):
    if isinstance(model_src, str):
        checkpoint = load_model(model_src, "cpu")
        summary(checkpoint['model'], input_size=input_size)
    elif isinstance(model_src, torch.nn.Module):
        summary(model_src, input_size=input_size)


def save_data(filename: str, data: np.ndarray|torch.Tensor) -> None:
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


def tuple2list(t: tuple):
    return list(t)


def list2tuple(l: list):
    return tuple(l)


def to_numpy(x: np.ndarray|torch.Tensor|list|tuple) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, tuple):
        return np.array(x)


def to_tensor(x: np.ndarray|torch.Tensor|list|tuple):
    if isinstance(x, np.ndarray):
        return torch.Tensor(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return torch.Tensor(x)
    if isinstance(x, tuple):
        return torch.Tensor(x)


# torch shape: (B, C, H, W)
# numpy shape: (C, H, W)
# PIL.Image shape: (W, H, C)
