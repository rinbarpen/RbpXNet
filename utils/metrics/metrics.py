from typing import List

import numpy as np

from .scores import accuracy_score, dice_score, iou_score, precision_score, recall_score, f1_score, roc_auc_score, auc


def get_metrics(targets: np.ndarray, preds: np.ndarray, labels: List[str],
                selected: List[str]
                =["accuracy", "mIoU", "recall", "precision", "f1", "dice", "roc", "auc"]):
    """
    This function calculates and returns a dictionary of selected metrics for a given set of targets and predictions.

    Parameters:
    - targets (np.array): A 2D numpy array representing the true labels of the data.
    - preds (np.array): A 2D numpy array representing the predicted labels of the data.
    - labels (List[str]): A list of unique class labels.
    - average (Literal["micro", "macro", "samples", "weighted", "binary", "binary"]): The averaging strategy for the metrics. Default is 'macro'.
    - selected (List[str]): A list of metric names to be calculated. Default is ["accuracy", "mIoU", "recall", "precision", "f1", "pa", "dice", "roc", "auc"].

    Returns:
    - results (dict): A dictionary containing the calculated metrics.
    """

    results = dict()
    if 'accuracy' in selected:
        results['accuracy'] = accuracy_score(targets, preds)
    if 'mIoU' in selected:
        results['mIoU'] = iou_score(targets, preds)
    if 'recall' in selected:
        results['recall'] = recall_score(targets, preds)
    if 'precision' in selected:
        results['precision'] = precision_score(targets, preds)
    if 'f1' in selected:
        results['f1'] = f1_score(targets, preds)
    if 'dice' in selected:
        results['dice'] = dice_score(targets, preds)
    if 'roc' in selected:
        results['roc'] = roc_auc_score(targets, preds)
    if 'auc' in selected:
        results['auc'] = auc(targets, preds)

    return results
