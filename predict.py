import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from utils.utils import load_model, create_dirs
from utils.visualization import draw_attention_heat_graph, draw_heat_graph


def predict_one(net, input: Path, classes: List[str], device):
    """
    This function performs a single prediction on an input image using a given neural network model.
    It resizes the input image, performs a forward pass through the network, and generates a prediction
    for each class. The function also converts the prediction to an image and resizes it to the original size.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    input (Union[str, Image.Image]): The input image. It can be either a path to an image file or a PIL Image object.
    classes (List[str]): A list of class names corresponding to the output of the neural network.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    Dict[str, Dict[str, np.ndarray, Image.Image]]: A dictionary containing the results of the prediction.
    The outer dictionary keys are the class names, and the values are dictionaries containing the prediction possibility
    as a numpy array and the corresponding image as a PIL Image object.
    """
    from config import CONFIG
    net.load_state_dict(load_model(CONFIG['load'], device=device)['model'])

    # input = Image.open(input).convert('RGB')
    # original_size = input.size # (W, H)
    # input = input.resize((512, 512))  # TODO: to be more flexible
    # input = torch.from_numpy(np.array(input).transpose(2, 0, 1))

    input = Image.open(input).convert('L')
    original_size = input.size
    input = input.resize((512, 512))
    input = torch.from_numpy(np.array(input))

    input = input.unsqueeze(0).unsqueeze(0) # (1, 1, 512, 512)
    net, input = net.to(device, dtype=torch.float32), input.to(device, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        predict = net(input)  # (1, N, H, W)

        predict = (predict - predict.min()) / (predict.max() - predict.min())
        predict = predict.squeeze(0)
        possibilities = predict.cpu().detach().numpy()  # (N, H, W)
        predict_image = possibilities.copy()  # (N, H, W)

        predict_image[predict_image >= 0.5] = 255
        predict_image[predict_image < 0.5] = 0
        predict_image = predict_image.astype(np.uint8)

    assert predict_image.shape[0] == len(classes), "The number of classes should be equal to the number of ones predicted"
    result = dict()
    for i, name in enumerate(classes):
        possibility = possibilities[i]
        mask = Image.fromarray(predict_image[i], mode='L').resize(size=original_size)
        result[name] = {
            "possibility": possibility,
            "mask": mask,
        }
    # background_mask = (1.0 - possibilities.max())
    # background_mask[background_mask >= 0.5] = 0
    # background_mask[background_mask < 0.5] = 255
    # result['background'] = {
    #     "possibility": 1.0 - possibilities.max(dim=0),
    #     "mask": Image.fromarray(background_mask, mode='L').resize(size=original_size)
    # }
    return result


def predict_more(net, inputs: List[Path], classes: List[str], device):
    """
    This function performs predictions on multiple input images using a given neural network model.
    It iterates over the input images, calls the `predict_one` function for each image, and stores
    the results in a dictionary.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    inputs (List[str]): A list of paths to the input image files.
    classes (List[str]): A list of class names corresponding to the output of the neural network.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    Dict[str, Dict[str, Dict[str, np.ndarray, Image.Image]]]: A dictionary containing the results of the predictions.
    The outer dictionary keys are the input image paths, and the values are dictionaries containing the results for each image.
    The inner dictionaries have class names as keys, and the values are dictionaries containing the prediction possibility
    as a numpy array and the corresponding image as a PIL Image object.
    """
    results = dict()
    for input in inputs:
        result = predict_one(net, input, classes=classes, device=device)
        results[input.absolute()] = result

    return results


def predict(net, inputs: List[Path], classes: List[str], device):
    """
    This function performs predictions on a list of input images using a given neural network model.
    It iterates over the input images, calls the `predict_one` function for each image, and generates
    heat maps and attention-based heat maps for each prediction.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    inputs (List[str]): A list of paths to the input image files.
    classes (List[str]): A list of class names corresponding to the output of the neural network.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    None. The function generates heat maps and attention-based heat maps for each prediction.
    """
    from config import CONFIG
    for input in inputs:
        result = predict_one(net, input, classes=classes, device=device)

        predict_dir = CONFIG["save"]["predict_dir"]
        filename = input.stem
        predict_filename = f"{filename}_predict.png"
        heat_filename = f"{filename}_heat.png"
        fuse_filename = f"{filename}_fuse.png"
        for category, values in result.items():
            category_dir = f"{predict_dir}{category}/"
            predict_path = f"{category_dir}{predict_filename}"
            heat_path = f"{category_dir}{heat_filename}"
            fuse_path = f"{category_dir}{fuse_filename}"

            create_dirs(category_dir)

            possibility = values["possibility"]
            mask_image = values["mask"]
            mask_image.save(predict_path)
            original_image = Image.open(input).convert('L')
            logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
            draw_heat_graph(possibility,
                            filename=heat_path,
                            title=filename)
            logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
            draw_attention_heat_graph(mask_image,
                                      original_image,
                                      filename=fuse_path,
                                      title=filename)
            logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")
