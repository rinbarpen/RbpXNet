import logging
import os
from typing import Union, Tuple, List

import numpy as np
import torch
from models.unet.unet import UNet, UNetOriginal


def create_file_unsafe(filename):
    with open(filename, 'w'):
        pass


def create_file(filename: str) -> None:
    """
    This function creates a new file with the given filename. If the file already exists, 
    it does nothing.

    Parameters:
    filename (str): The name of the file to be created. The path to the file can be included.

    Returns:
    None
    """
    if os.path.exists(filename):
        return

    create_file_unsafe(filename)


def create_dirs(path: str) -> None:
    """
    This function creates a directory at the specified path if it does not already exist.

    Parameters:
    path (str): The path to the directory to be created. If the path includes parent directories, 
                they will also be created if they do not exist.

    Returns:
    None
    """
    if os.path.exists(path):
        return

    os.makedirs(path)


def create_file_parents(filename: str) -> None:
    """
    This function creates the parent directories of the specified file if they do not exist.
    If the file itself already exists or the parent directories exist, it does nothing.

    Parameters:
    filename (str): The name of the file for which the parent directories need to be created.
                    The path to the file can be included.

    Returns:
    None
    """
    dirname = os.path.dirname(filename)
    if os.path.exists(filename) or os.path.exists(dirname):
        return
    os.makedirs(dirname)


def create_file_if_not_exist(filename: str) -> None:
    """
    This function creates a new file with the given filename if it does not already exist.
    If the file already exists, it does nothing. If the parent directories do not exist, 
    they will be created.

    Parameters:
    filename (str): The name of the file to be created. The path to the file can be included.

    Returns:
    None
    """
    try:
        create_file_unsafe(filename)
    except FileNotFoundError:
        create_file_parents(filename)
        create_file_unsafe(filename)


def file_prefix_name(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[0]

def file_suffix_name(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[1]

def save_model(filename, model, optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    """
    This function saves the state dictionary of a PyTorch model to a file.
    If the file does not exist, it will be created. If the file exists, the existing file will be overwritten.
    If the parent directories do not exist, they will be created.

    Parameters:
    model (torch.nn.Module): The PyTorch model to save.
    filename (str): The name of the file to save the model state dictionary to. The path to the file can be included.
    optimizer (torch.optim.Optimizer, optional): The optimizer used for training the model. Defaults to None.
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler used for training the model. Defaults to None.
    scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler used for training the model. Defaults to None.
    **kwargs: Additional keyword arguments to be saved in the checkpoint.

    Returns:
    None
    """
    try:
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

        torch.save(checkpoint, filename)
    except FileNotFoundError:
        create_file_parents(filename)
        torch.save(model.state_dict(), filename)


def load_model(filename: str, device: torch.device) -> dict:
    """
    This function loads a PyTorch model's state dictionary from a file.

    Parameters:
    filename (str): The name of the file to load the model state dictionary from.
                    The path to the file can be included.
    device (torch.device): The device where the model will be loaded. This is used to map the model's state dictionary to the device.

    Returns:
    dict: The loaded model's state dictionary.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    Exception: If any other error occurs while loading the model.
    """
    try:
        checkpoint = torch.load(filename, map_location=device)
        return checkpoint
    except FileNotFoundError as e:
        logging.error(f'File Not Found: {e}')
        raise e
    except Exception as e:
        logging.error(f'Error loading model: {e}')
        raise e


def save_data(filename: str, data: Union[np.ndarray, torch.Tensor]) -> None:
    """
    This function saves a NumPy array or PyTorch tensor to a file in .npy format.
    If the input data is a PyTorch tensor, it will be converted to a NumPy array before saving.
    If the file does not exist, it will be created. If the file exists, the existing file will be overwritten.
    If the parent directories do not exist, they will be created.

    Parameters:
    filename (str): The name of the file to save the data to. The path to the file can be included.
    data (Union[np.ndarray, torch.Tensor]): The data to be saved. It can be either a NumPy array or a PyTorch tensor.

    Returns:
    None
    """
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


def select_model(model: str, *args, **kwargs):
    match (model):
        case 'UNet':
            return UNet(kwargs['in_channels'], kwargs['n_classes'], kwargs['use_bilinear'])
        case 'UNetOriginal':
            return UNetOriginal(kwargs['in_channels'], kwargs['n_classes'], kwargs['use_bilinear'])
        case _:
            raise ValueError(f'Not supported model: {model}')


def do_if(condition, fn, *args, **kwargs):
    if condition:
        return fn(args, kwargs)
    return None

def do_if_not(condition, fn, *args, **kwargs):
    if not condition:
        return fn(args, kwargs)
    return None

def where(condition, true_fn, false_fn):
    return true_fn() if condition else false_fn()

