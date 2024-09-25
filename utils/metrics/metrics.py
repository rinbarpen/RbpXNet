from typing import List, Union, Literal, Optional

import numpy as np

from utils.metrics.scores import dice_score, iou_score, precision_score, recall_score, f1_score, accuracy_score
from utils.visualization import draw_metrics_graph


def get_metrics(targets: np.ndarray, preds: np.ndarray, labels: List[str],
                selected: List[str]
                =["accuracy", "mIoU", "recall", "precision", "f1", "dice"]):
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

class MetricRecorder:
    def __init__(self, **kwargs):
        super(MetricRecorder, self).__init__()

        self.metrics = dict()

    def record(self, name: str, f, *args):
        self.metrics[name] = f(*args)
        return self

    def show(self):
        return MetricShower(self)

class MetricShower:
    def __init__(self, recorder):
        super(MetricShower, self).__init__()
        self.recorder = recorder

    def get_metric(self, name: str, mode: Literal['mean', 'sum', 'all']):
        if mode == 'mean':
            return np.mean(self.recorder.metrics[name])
        elif mode == 'all':
            return np.ndarray(self.recorder.metrics[name])
        elif mode == 'sum':
            return np.sum(self.recorder.metrics[name])

    def get_all_metrics(self):
        return self.recorder.metrics

    def draw_graph(self, metrics_select: List[str], filename: Optional[str] = None):
        COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "brown"]
        draw_metrics_graph(self.recorder.metrics, colors=COLORS, filename=filename, selected=metrics_select, title="Metrics")
