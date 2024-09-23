import numpy as np

import metrics

def dice_loss(inputs: np.ndarray, targets: np.ndarray, num_classes :int):
    return 1.0 - metrics.dice_score(inputs, targets, num_classes)['average']
