import os
from typing import Union, List, Dict
import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from utils.utils import load_model
import numpy as np

from utils.visualization import draw_attention_heat_graph, draw_heat_graph

def predict_one(net, input: Union[str, Image.Image], device) -> Tuple[torch.Tensor, Image.Image]:
    """
    This function takes a neural network model, an input image, and a device,
    performs a prediction on the image, and returns the prediction and the corresponding grayscale image.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    input (Union[str, Image.Image]): The input image. It can be either a path to the image file or a PIL Image object.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    Tuple[torch.Tensor, Image.Image]: A tuple containing the prediction (a torch.Tensor) and the corresponding grayscale image (a PIL Image object).
    """
    net.load_state_dict(load_model(wandb.config.load))
    if isinstance(input, str):
        input = Image.open(input).convert('RGB')
    
    original_size = input.size
    input = input.resize((512, 512))  # TODO: to be more flexible
    input = torch.from_numpy(np.array(input))
    
    input = input.expand(1, -1, -1, -1).permute(0, 3, 1, 2)
    net, input = net.to(device, dtype=torch.float32), input.to(device, dtype=torch.float32)
    
    net.eval()
    with torch.no_grad():
        predict = net(input)
        predict_np = predict.squeeze().cpu().detach().numpy()
        predict_np[predict_np >= 0.5] = 255
        predict_np[predict_np < 0.5] = 0
    
    predict_image = Image.fromarray(predict_np).convert('L')
    predict_image.resize(size=original_size)
    return predict, predict_image 


def predict_more(net, inputs, device):
    """
    This function takes a neural network model, a list of input images, and a device,
    and performs predictions on the images. It collects the predictions and their corresponding
    images, returning them as lists.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    inputs (List[Union[str, Image.Image]]): A list of input images, which can be either paths to the image files or PIL Image objects.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    Tuple[List[torch.Tensor], List[Image.Image]]: A tuple containing two lists. The first list contains the predictions,
    and the second list contains the corresponding images.
    """
    predicts, predict_images = [], []
    for input in inputs:
        predict, predict_image = predict_one(net, input, device)
        predicts.append(predict)
        predict_images.append(predict_image)
    
    return predicts, predict_images

def predict(net, inputs: List[str], device):
    """
    This function takes a neural network model, a list of input image paths, and a device,
    and performs predictions on the images. It then generates heatmaps and attention-based heatmaps
    for each prediction, saving them as PNG files.

    Parameters:
    net (torch.nn.Module): The neural network model to be used for predictions.
    inputs (List[str]): A list of paths to the input images.
    device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    None. The function saves the generated heatmaps and attention-based heatmaps as PNG files.
    """
    for input in inputs:
        predict, predict_image = predict_one(net, input, device)
        
        predict_filename = os.path.splitext(input)[0] + "_predict" + ".png"
        fuse_filename = os.path.splitext(input)[0] + "_fuse" + ".png"
        
        draw_heat_graph(predict, 
                        filename=predict_filename, 
                        title=os.path.splitext(os.path.basename(input))[0])
        draw_attention_heat_graph(predict, predict_image, 
                                  filename=fuse_filename,
                                  title=os.path.splitext(os.path.basename(input))[0])
    