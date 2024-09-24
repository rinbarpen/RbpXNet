import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from utils.utils import load_model, create_dirs
from utils.visualization import draw_attention_heat_graph, draw_heat_graph


@torch.inference_mode()
def predict_one(net, input: Path, classes: List[str], device):
    from config import CONFIG
    net.load_state_dict(load_model(CONFIG['load'], device=device)['model'])

    # input = Image.open(input).convert('RGB')
    # original_size = input.size # (W, H)
    # input = input.resize((512, 512))  # TODO: to be more flexible
    # input = torch.from_numpy(np.array(input).transpose(2, 0, 1))
    # input = input.unsqueeze(0) # (1, 3, 512, 512)

    input = Image.open(input).convert('L')
    original_size = input.size
    input = input.resize((512, 512))
    input = torch.from_numpy(np.array(input))

    input = input.unsqueeze(0).unsqueeze(0) # (1, 1, 512, 512)
    net, input = net.to(device, dtype=torch.float32), input.to(device, dtype=torch.float32)

    net.eval()
    
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
    results = dict()
    for input in inputs:
        result = predict_one(net, input, classes=classes, device=device)
        results[input.absolute()] = result

    return results


def predict(net, inputs: List[Path], classes: List[str], device):
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
            original_image = Image.open(input).convert('RGB')
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
