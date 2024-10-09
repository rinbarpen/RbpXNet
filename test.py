import logging
import os
from typing import *

import torch
from tqdm import tqdm

from utils.metrics.metrics import get_metrics
from utils.utils import load_model
from utils.Recorder import Recorder


class Tester:
    def __init__(self, net, loader, classes: List[str], device=None):
        from config import CONFIG
        
        self.device = device if device else CONFIG['device']
        self.net = net.to(self.device)
        self.loader = loader
        self.classes = classes
        
        assert len(self.classes) >= 1, 'The number of classes should be greater than 0'

    @torch.no_grad()
    def test_model(self, selected_metrics: List[str] = ["dice", "f1", "recall"]):
        self.net.eval()
        
        all_metrics = dict()
        mean_metrics = dict()
        with tqdm(total=len(self.loader), desc=f'Testing') as pbar:
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(self.device, dtype=torch.float32), targets.to(self.device, dtype=torch.float32)

                preds = self.net(inputs)

                threshold = 0.5
                preds[preds >= threshold] = 1
                preds[preds < threshold] = 0

                metrics = get_metrics(
                    targets.cpu().detach().numpy(),
                    preds.cpu().detach().numpy(),
                    labels=self.classes,
                    selected=selected_metrics
                )

                for k, v in metrics.items():
                    all_metrics[k] = v['all']
                    if k in mean_metrics:
                        mean_metrics[k] += v['mean']
                    else:
                        mean_metrics[k] = v['mean']
                
                for k, v in mean_metrics.items():
                    mean_metrics[k] /= len(self.loader)
                
                pbar.update()
                pbar.set_postfix(**{
                    'accuracy': metrics['accuracy']['mean']
                })

        return all_metrics, mean_metrics

    def test(self, selected_metrics: List[str]):
        from config import CONFIG

        model_filename = CONFIG['save']['model']
        logging.info(f"Loading model: {os.path.abspath(model_filename)} on {self.device}")
        
        self.net.load_state_dict(load_model(model_filename, self.device)["model"])
        all_metrics, mean_metrics = self.test_model(selected_metrics=selected_metrics)

        Recorder.record_test(Recorder(), 'metric'=mean_metrics)
