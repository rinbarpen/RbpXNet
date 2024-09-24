import logging
import os
from typing import *

import torch
from tqdm import tqdm

from utils.metrics.metrics import get_metrics
from utils.utils import load_model
from utils.visualization import draw_metrics_graph
from utils.writer import CSVWriter


@torch.no_grad()
def test_model(net, device, test_loader,
               classes: List[str], selected_metrics: List[str] = ["dice", "f1", "recall"]):
    assert len(classes) >= 1, 'predict the number of classes should be greater than 0'

    net.to(device)
    net.eval()

    mean_metrics = dict()
    n_step = len(test_loader)
    with tqdm(total=n_step, desc=f'Testing') as pbar:
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)

            preds = net(inputs)

            threshold = 0.5
            preds[preds >= threshold] = 1
            preds[preds < threshold] = 0

            metrics = get_metrics(
                targets.cpu().detach().numpy(),
                preds.cpu().detach().numpy(),
                labels=classes,
                selected=selected_metrics
            )

            for k, v in metrics.items():
                if k in mean_metrics:
                    mean_metrics[k] += v['mean']
                else:
                    mean_metrics[k] = v['mean']

            pbar.update()
            pbar.set_postfix(**{
                'accuracy': metrics['accuracy']['mean']
            })

    for k in mean_metrics.keys():
        mean_metrics[k] /= len(test_loader)

    return mean_metrics


def test(net, test_loader, device, classes: List[str], selected_metrics: List[str]):
    from config import CONFIG

    net.to(device)

    model_filename = CONFIG['save']['model']
    logging.info(f"Loading model: {os.path.abspath(model_filename)} on {device}")
    net.load_state_dict(load_model(model_filename, device)["model"])
    metrics = test_model(net,
                         device=device,
                         test_loader=test_loader,
                         classes=classes,
                         selected_metrics=selected_metrics)

    test_csv_filename = f"{CONFIG['save']['test_dir']}test_metrics.csv"
    writer = CSVWriter(test_csv_filename)
    (writer.write_headers(list(metrics.keys()))
     .writes(metrics)
     .flush())
    logging.info(f"Save metrics data to {os.path.abspath(test_csv_filename)}")

    test_loss_image_path = f"{CONFIG['save']['test_dir']}metrics.png"
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    draw_metrics_graph(metrics,
                       title='Metrics',
                       colors=colors,
                       selected=list(metrics.keys()),
                       filename=test_loss_image_path)
    logging.info(f"Save metrics graph to {os.path.abspath(test_loss_image_path)}")
    if CONFIG['wandb']:
        import wandb
        wandb.log({'metrics': metrics, 'metrics_image': test_loss_image_path})
