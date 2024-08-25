import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from utils.utils import load_model
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
    net.load_state_dict(load_model(CONFIG['load'], device=device))

    input = Image.open(input).convert('L')
    original_size = input.size

    input = input.resize((512, 512))  # TODO: to be more flexible
    input = torch.from_numpy(np.array(input))

    input = input.unsqueeze(0).unsqueeze(0)
    net, input = net.to(device, dtype=torch.float32), input.to(device, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        predict = net(input)  # (1, N, H, W)
        possibilities = predict.squeeze(0).cpu().detach().numpy()  # (N, H, W)
        predict_np = possibilities.copy()  # (N, H, W)
        predict_np[predict_np >= 0.5] = 255
        predict_np[predict_np < 0.5] = 0

    assert predict_np.shape[0] == len(classes), "The number of classes should be equal to the number of ones predicted"
    result = dict()
    for i, name in enumerate(classes):
        predict_image = cv2.resize(predict_np[i],
                                   (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
        result[name] = {
            "possibility": possibilities[i],
            "image": predict_image,
        }
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

        for category, values in result.items():
            predict_dir = CONFIG["save"]["predict_dir"]
            filename = input.stem
            predict_filename = f"{filename}_{category}_predict.png"
            heat_filename = f"{filename}_{category}_heat.png"
            fuse_filename = f"{filename}_{category}_fuse.png"
            predict_path = f"{predict_dir}{predict_filename}"
            heat_path = f"{predict_dir}{heat_filename}"
            fuse_path = f"{predict_dir}{fuse_filename}"

            cv2.imwrite(predict_path, values["image"])
            # TODO:
            logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
            draw_heat_graph(values["possibility"],
                            filename=heat_path,
                            title=filename)
            logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
            draw_attention_heat_graph(values["image"],
                                      Image.open(input).convert('RGB'),
                                      filename=fuse_path,
                                      title=filename)
            logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")
