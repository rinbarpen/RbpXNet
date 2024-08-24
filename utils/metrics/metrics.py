from typing import List, Literal

import numpy as np

from .accuracy_score import accuracy_score
from .dice_score import dice_score
from .f1_score import f1_score
from .iou_score import iou_score
from .pa_score import pa_score
from .precision_score import precision_score
from .recall_score import recall_score
from .roc_auc_score import roc_auc_score, auc


def get_metrics(targets: np.array, preds: np.array, labels: List[str], 
                average: Literal["micro", "macro", "samples", "weighted", "binary", "binary"]
                ='macro', 
                selected: List[str]
                =["accuracy", "mIoU", "recall", "precision", "f1", "pa", "dice", "roc", "auc"]):
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
        results['accuracy'] = accuracy_score(targets, preds, labels)
    if 'mIoU' in selected:
        results['mIoU'] = iou_score(targets, preds, labels)[1]
    if 'recall' in selected:
        results['recall'] = recall_score(targets, preds, labels=labels)
    if 'precision' in selected:
        results['precision'] = precision_score(targets, preds, labels=labels)
    if 'f1' in selected:
        results['f1'] = f1_score(targets, preds, labels=labels)
    if 'pa' in selected:
        results['pa'] = pa_score(targets, preds, labels)[1]
    if 'dice' in selected:
        results['dice'] = dice_score(targets, preds, labels)[1]
    if 'roc' in selected:
        results['roc'] = roc_auc_score(targets, preds, labels)
    if 'auc' in selected:
        results['auc'] = auc(targets, preds, labels)

    return results
