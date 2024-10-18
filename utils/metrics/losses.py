import numpy as np


def dice_loss(inputs: np.ndarray, targets: np.ndarray, smooth: float=1e-5):
    scores = 0.0
    n_classes = inputs.shape[1]
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (inputs == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        dice = (2.0 * intersection + smooth) / (intersection + union + smooth)
        scores += dice

    return 1.0 - scores / n_classes


def focal_loss(inputs: np.ndarray, targets: np.ndarray, smooth: float=1e-5):
    scores = 0.0
    n_classes = inputs.shape[1]
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (inputs == i)

        TP = np.sum(pred_binary * target_binary)
        FP = np.logical_and(pred_binary == 1, target_binary == 0).sum()

        precision = (TP + smooth) / (TP + FP + smooth)
        scores += precision

    return 1.0 - scores / n_classes
