from typing import List

import numpy as np

from .scores import dice_score, iou_score, precision_score, recall_score, f1_score, accuracy_score


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
