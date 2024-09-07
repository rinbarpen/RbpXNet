import numpy as np
# from sklearn import metrics as sk_metrics
# from sklearn.metrics import f1_score


# def _preprocess_data(targets: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Preprocess the input data for evaluation metrics.
#
#     This function takes in the ground truth targets and predicted probabilities,
#     reshapes them into 2D arrays, and then extracts the class labels.
#
#     Parameters:
#     - targets (np.ndarray): A 4D numpy array representing the ground truth targets.
#         The shape should be (N, C, H, W), where N is the number of samples,
#         C is the number of classes, H is the height, and W is the width.
#     - preds (np.ndarray): A 4D numpy array representing the predicted probabilities.
#         The shape should be (N, C, H, W), where N is the number of samples,
#         C is the number of classes, H is the height, and W is the width.
#
#     Returns:
#     - targets_labels (np.ndarray): A 1D numpy array representing the ground truth class labels.
#         The shape is (N * H * W,).
#     - preds_labels (np.ndarray): A 1D numpy array representing the predicted class labels.
#         The shape is (N * H * W,).
#     """
#     assert targets.shape == preds.shape, "targets.shape and preds.shape do not match"
#     N, C, H, W = targets.shape
#     targets_flat = targets.transpose(0, 2, 3, 1).reshape(N * H * W, C)
#     preds_flat = preds.transpose(0, 2, 3, 1).reshape(N * H * W, C)
#
#     targets_labels = np.argmax(targets_flat, axis=1)
#     preds_labels = np.argmax(preds_flat, axis=1)
#
#     return targets_labels, preds_labels


# def iou_score(targets: np.ndarray, preds: np.ndarray, average="macro"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#
#     cm = sk_metrics.confusion_matrix(targets_labels, preds_labels)
#
#     # IoU 是 TP / (TP + FP + FN) 对于每个类别
#     intersection = np.diag(cm)  # 对角线上是每个类的 TP
#     union = cm.sum(axis=1) + cm.sum(axis=0) - intersection  # TP + FP + FN
#
#     # 计算每个类的 IoU
#     iou = intersection / union if union != 0 else 0
#
#     if average == "macro":
#         return np.mean(iou)
#     elif average == "micro":
#         return np.sum(intersection) / np.sum(union)
#     elif average is None:
#         return iou
#     else:
#         raise ValueError(f"Unknown average type: {average}")

# def accuracy_score(targets: np.ndarray, preds: np.ndarray):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#
#     return sk_metrics.accuracy_score(targets_labels, preds_labels)
#
# def precision_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     return sk_metrics.precision_score(targets_labels, preds_labels, average=average)
#
# def f1_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     return sk_metrics.f1_score(targets_labels, preds_labels, average=average)
#
# def recall_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     return sk_metrics.recall_score(targets_labels, preds_labels, average=average)
#
#
# def dice_score(targets: np.ndarray, preds: np.ndarray, average="binary"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     return sk_metrics.f1_score(targets_labels, preds_labels, average=average)
#
# def roc_curve(targets: np.ndarray, preds: np.ndarray):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     fpr, tpr, thresholds = sk_metrics.roc_curve(targets_labels, preds_labels, pos_label=1)
#     return fpr, tpr, thresholds
#
# def roc_auc_score(targets: np.ndarray, preds: np.ndarray, average="macro"):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#     return sk_metrics.roc_auc_score(targets_labels, preds_labels, average=average)
#
# def auc(targets: np.ndarray, preds: np.ndarray):
#     targets_labels, preds_labels = _preprocess_data(targets, preds)
#
#     fpr, tpr, _ = sk_metrics.roc_curve(targets_labels, preds_labels)
#
#     return sk_metrics.auc(fpr, tpr)

# def fast_hist(targets: np.ndarray, preds: np.ndarray, n_classes: int):
#     hist = np.zeros((n_classes, n_classes))
#     k = (targets >= 0) & (targets < n_classes)
#
#     return (np.bincount(n_classes *
#                         targets[k].astype(np.int32) +
#                         preds[k],
#                         minlength=n_classes ** 2)
#             .reshape(n_classes, n_classes))
#
#
# def per_class_iou_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
#     hist = fast_hist(targets, preds, n_classes)
#     return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)
#
# def per_class_recall_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
#     hist = fast_hist(targets, preds, n_classes)
#     return np.diag(hist) / np.maximum(hist.sum(1), 1)
#
# def per_class_precision_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
#     hist = fast_hist(targets, preds, n_classes)
#     return np.diag(hist) / np.maximum(hist.sum(0), 1)
#
# def per_class_accuracy_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
#     hist = fast_hist(targets, preds, n_classes)
#     return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def iou_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
    scores = []
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        iou = intersection / union if union > 0 else 0.0
        scores.append(iou)

    return {
        "all": scores,
        "average": np.mean(scores)
    }

def dice_score(targets: np.ndarray, preds: np.ndarray, n_classes: int):
    dice_scores = []

    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        dice = 2.0 * intersection / union if union > 0 else 0.0
        dice_scores.append(dice)

    return {
        "all": dice_scores,
        "average": np.mean(dice_scores),
    }

def f1_score(targets: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """
    Calculates the F1 score for each class in a semantic segmentation task.

    The F1 score is calculated as the harmonic mean of precision and recall.
    This function iterates over each class, converts the target and prediction arrays to binary arrays,
    and then calculates the F1 score for each class.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - preds (np.ndarray): A 4D numpy array representing the predicted labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - n_classes (int): The total number of classes in the dataset.

    Returns:
    - dict: A dictionary containing the F1 scores for each class and the average score.
        The keys are 'all' and 'average'.
        The value for 'all' is a list of F1 scores for each class.
        The value for 'average' is the average F1 score across all classes.
    """
    scores = []

    for i in range(n_classes):
        pred_binary = (preds == i).astype(np.float32)
        target_binary = (targets == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP
        FN = np.sum(target_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    return {
        "all": scores,
        "average": np.mean(scores),
    }

def recall_score(targets: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """
    Calculates the recall score for each class in a semantic segmentation task.

    The recall score is calculated as the ratio of true positives to the sum of true positives and false negatives.
    This function iterates over each class, converts the target and prediction arrays to binary arrays,
    and then calculates the recall score for each class.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - preds (np.ndarray): A 4D numpy array representing the predicted labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - n_classes (int): The total number of classes in the dataset.

    Returns:
    - dict: A dictionary containing the recall scores for each class and the average score.
        The keys are 'all' and 'average'.
        The value for 'all' is a list of recall scores for each class.
        The value for 'average' is the average recall score across all classes.
    """
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FN = np.sum(target_binary) - TP

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        scores.append(recall)

    return {
        "all": scores,
        "average": np.mean(scores),
    }

def accuracy_score(targets: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """
    Calculates the accuracy score for each class in a semantic segmentation task.

    The accuracy score is calculated as the ratio of correctly predicted samples to the total samples.
    This function iterates over each class, converts the target and prediction arrays to binary arrays,
    and then calculates the accuracy score for each class.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - preds (np.ndarray): A 4D numpy array representing the predicted labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - n_classes (int): The total number of classes in the dataset.

    Returns:
    - dict: A dictionary containing the accuracy scores for each class and the average score.
        The keys are 'all' and 'average'.
        The value for 'all' is a list of accuracy scores for each class.
        The value for 'average' is the average accuracy score across all classes.
    """
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        correct = np.sum(pred_binary * target_binary)
        total = pred_binary.size
        
        accuracy = correct / total if total > 0 else 0.0
        scores.append(accuracy)

    return {
        "all": scores,
        "average": np.mean(scores),
    }


def precision_score(targets: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """
    Calculates the precision score for each class in a semantic segmentation task.

    The precision score is calculated as the ratio of true positives to the sum of true positives and false positives.
    This function iterates over each class, converts the target and prediction arrays to binary arrays,
    and then calculates the precision score for each class.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - preds (np.ndarray): A 4D numpy array representing the predicted labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - n_classes (int): The total number of classes in the dataset.

    Returns:
    - dict: A dictionary containing the precision scores for each class and the average score.
        The keys are 'all' and 'average'.
        The value for 'all' is a list of precision scores for each class.
        The value for 'average' is the average precision score across all classes.
    """
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        scores.append(precision)

    return {
        "all": scores,
        "average": np.mean(scores),
    }

def scores(targets: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """
    Calculates various evaluation metrics for semantic segmentation predictions.

    Parameters:
    - targets (np.ndarray): A 4D numpy array representing the ground truth labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - preds (np.ndarray): A 4D numpy array representing the predicted labels.
        The shape should be (B, C, H, W), where B is the batch size,
        C is the number of classes, H is the height, and W is the width.
        The values should be integers in the range [0, n_classes).
    - n_classes (int): The total number of classes in the dataset.

    Returns:
    - dict: A dictionary containing the calculated evaluation metrics.
        The keys are 'miou', 'dice', 'f1', 'recall', and 'precision'.
        The values are dictionaries containing the per-class scores and the average score.
    """
    
    ious = iou_score(targets, preds, n_classes)
    dice = dice_score(targets, targets, n_classes)
    recall = recall_score(targets, preds, n_classes)
    f1 = f1_score(targets, targets, n_classes)
    precision = precision_score(targets, preds, n_classes)
    accuracy = accuracy_score(targets, preds, n_classes)

    return {
        "miou": ious,
        "dice": dice,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }
