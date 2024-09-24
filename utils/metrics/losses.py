import numpy as np

from utils.metrics.scores import dice_score, focal_score


def dice_loss(inputs: np.ndarray, targets: np.ndarray, num_classes: int, smooth: float=1e-5):
    return 1.0 - dice_score(targets, inputs, num_classes, smooth)['mean']

def focal_loss(inputs: np.ndarray, targets: np.ndarray, num_classes: int, smooth: float=1e-5):
    return 1.0 - focal_score(targets, inputs, num_classes, smooth)['mean']
