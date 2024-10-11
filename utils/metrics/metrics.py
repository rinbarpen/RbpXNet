from typing import List, Union, Literal, Optional

import numpy as np
from copy import deepcopy, copy
from defines import AllScoreDict
from utils.metrics.scores import dice_score, iou_score, precision_score, recall_score, f1_score, accuracy_score
from utils.visualization import draw_metrics_graph

class MetricRecorder:
    def __init__(self, **kwargs):
        super(MetricRecorder, self).__init__()

        self.metrics = dict(**kwargs)

    def record(self, name: str, values: dict):
        for label, value in values.items():
            self.metrics[name][label].extend(value)
        return self

    def filter(self, names: tuple):
        filtered_metrics = dict()
        for selected in filter(lambda x: x in names, self.metrics.keys()):
            filtered_metrics[selected] = self.metrics[selected]
        return MetricRecorder(**filtered_metrics)

    def view(self):
        return MetricViewer(self)

class MetricViewer:
    def __init__(self, recorder):
        super(MetricViewer, self).__init__()
        self.recorder = recorder

    def metric(self, name: str, label: str, mode: Literal['mean', 'std', 'all']):
        if mode == 'mean':
            return np.mean(self.recorder.metrics[name][label])
        elif mode == 'std':
            return np.std(self.recorder.metrics[name][label])
        elif mode == 'all':
            return np.ndarray(self.recorder.metrics[name][label])

    def all_metrics(self, mode: Literal['mean', 'std', 'all']='all'):
        metrics = dict()
        match mode:
            case 'all': 
                metrics = self.recorder.metrics
            case 'mean':
                for name, values in self.recorder.metrics.items():
                    for label in values.keys():
                        metrics[name][label] = self.metric(name, label, 'mean')
                for name, values in metrics.items():
                    mean = np.mean([x for x in values.values()])
                    metrics[name]['mean'] = mean 
            case 'std':
                for name, values in self.recorder.metrics.items():
                    for label in values.keys():
                        metrics[name][label] = self.metric(name, label, 'std')
                for name, values in metrics.items():
                    std = np.std([x for x in values.values()])
                    metrics[name]['std'] = std

        return metrics

    def draw_graph(self, metrics_select: List[str], filename: Optional[str] = None):
        COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "brown"]
        draw_metrics_graph(self.recorder.metrics, colors=COLORS, filename=filename, selected=metrics_select, title="Metrics")
