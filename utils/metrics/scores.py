from typing import Tuple

import numpy as np
from sklearn import metrics as sk_metrics


def _preprocess_data(targets: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the input data for evaluation metrics.

    This function takes in the ground truth targets and predicted probabilities,
    reshapes them into 2D arrays, and then extracts the class labels.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth targets.
        The shape should be (N, C, H, W), where N is the number of samples,
        C is the number of classes, H is the height, and W is the width.
    - preds (np.ndarray): A 4D numpy array representing the predicted probabilities.
        The shape should be (N, C, H, W), where N is the number of samples,
        C is the number of classes, H is the height, and W is the width.

    Returns:
    - targets_labels (np.ndarray): A 1D numpy array representing the ground truth class labels.
        The shape is (N * H * W,).
    - preds_labels (np.ndarray): A 1D numpy array representing the predicted class labels.
        The shape is (N * H * W,).
    """
    assert targets.shape == preds.shape, "targets.shape and preds.shape do not match"
    N, C, H, W = targets.shape
    targets_flat = targets.reshape(N * H * W, C)
    preds_flat = preds.reshape(N * H * W, C)

    targets_labels = np.argmax(targets_flat, axis=1)
    preds_labels = np.argmax(preds_flat, axis=1)

    return targets_labels, preds_labels


def iou_score(targets: np.ndarray, preds: np.ndarray, average="macro"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)

    cm = sk_metrics.confusion_matrix(targets_labels, preds_labels)

    # IoU 是 TP / (TP + FP + FN) 对于每个类别
    intersection = np.diag(cm)  # 对角线上是每个类的 TP
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection  # TP + FP + FN

    # 计算每个类的 IoU
    iou = intersection / union if union != 0 else 0

    if average == "macro":
        return np.mean(iou)
    elif average == "micro":
        return np.sum(intersection) / np.sum(union)
    elif average is None:
        return iou
    else:
        raise ValueError(f"Unknown average type: {average}")

def accuracy_score(targets: np.ndarray, preds: np.ndarray):
    targets_labels, preds_labels = _preprocess_data(targets, preds)

    return sk_metrics.accuracy_score(targets_labels, preds_labels)

def precision_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    return sk_metrics.precision_score(targets_labels, preds_labels, average=average)

def f1_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    return sk_metrics.f1_score(targets_labels, preds_labels, average=average)

def recall_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    return sk_metrics.recall_score(targets_labels, preds_labels, average=average)


def dice_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    return sk_metrics.f1_score(targets_labels, preds_labels, average=average)

def roc_curve(targets: np.ndarray, preds: np.ndarray):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    fpr, tpr, thresholds = sk_metrics.roc_curve(targets_labels, preds_labels, pos_label=1)
    return fpr, tpr, thresholds

def roc_auc_score(targets: np.ndarray, preds: np.ndarray, average="macro"):
    targets_labels, preds_labels = _preprocess_data(targets, preds)
    return sk_metrics.roc_auc_score(targets_labels, preds_labels, average=average)

def auc(targets: np.ndarray, preds: np.ndarray):
    targets_labels, preds_labels = _preprocess_data(targets, preds)

    fpr, tpr, _ = sk_metrics.roc_curve(targets_labels, preds_labels)

    return sk_metrics.auc(fpr, tpr)
