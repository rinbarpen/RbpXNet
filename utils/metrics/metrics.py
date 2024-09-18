from typing import List, Union, Literal, Optional

import numpy as np

from .scores import dice_score, iou_score, precision_score, recall_score, f1_score, accuracy_score
from ..visualization import draw_metrics_graph


def get_metrics(targets: np.ndarray, preds: np.ndarray, labels: List[str],
                selected: List[str]
                =["accuracy", "mIoU", "recall", "precision", "f1", "dice"]):
    """
    This function calculates and returns a dictionary of selected metrics for a given set of targets and predictions.

    Parameters:
    - targets (np.array): A 2D numpy array representing the true labels of the data.
    - preds (np.array): A 2D numpy array representing the predicted labels of the data.
    - labels (List[str]): A list of unique class labels.
    - average (Literal["micro", "macro", "samples", "weighted", "binary", "binary"]): The averaging strategy for the metrics. Default is 'macro'.
    - selected (List[str]): A list of metric names to be calculated. Default is ["accuracy", "mIoU", "recall", "precision", "f1", "pa", "dice"].

    Returns:
    - results (dict): A dictionary containing the calculated metrics.
    """

    n_classes = len(labels)
    results = dict()
    if 'accuracy' in selected:
        results['accuracy'] = accuracy_score(targets, preds, n_classes)
    if 'mIoU' in selected:
        results['mIoU'] = iou_score(targets, preds, n_classes)
    if 'recall' in selected:
        results['recall'] = recall_score(targets, preds, n_classes)
    if 'precision' in selected:
        results['precision'] = precision_score(targets, preds, n_classes)
    if 'f1' in selected:
        results['f1'] = f1_score(targets, preds, n_classes)
    if 'dice' in selected:
        results['dice'] = dice_score(targets, preds, n_classes)

    return results

class MetricRecoder:
    def __init__(self, **kwargs):
        super(MetricRecoder, self).__init__()

        self.metrics = dict()

    def add_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = [value]
        else:
            self.metrics[name].append(value)

        return self

    def get_metric(self, name: str, mode: str=Literal['mean', 'sum', 'all'], dtype=np.float32):
        if mode == 'mean':
            return np.mean(self.metrics[name], dtype=dtype)
        elif mode == 'all':
            return np.ndarray(self.metrics[name], dtype=dtype)
        elif mode == 'sum':
            return np.sum(self.metrics[name], dtype=dtype)

    def get_all_metrics(self):
        return self.metrics

    def draw_graph(self, metrics_select: List[str], filename: Optional[str] = None):
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "brown"]
        draw_metrics_graph(self.metrics, colors=colors, filename=filename, selected=metrics_select, title="Metrics")
