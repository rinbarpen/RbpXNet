import numpy as np

from defines import AllScoreDict, ScoreDict


def iou_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary) - intersection

        iou = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
        scores.append(iou)

    return {
        "all": scores,
        "mean": np.mean(scores)
    }

def dice_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    dice_scores = []

    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary) - intersection

        dice = (2.0 * intersection + smooth) / (union + intersection + smooth) if (union + intersection) > 0 else 0.0
        dice_scores.append(dice)

    return {
        "all": dice_scores,
        "mean": np.mean(dice_scores),
    }


def f1_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        pred_binary = (preds == i).astype(np.float32)
        target_binary = (targets == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP
        FN = np.sum(target_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = 2.0 * (precision * recall + smooth) / (precision + recall + smooth) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

# f score is defined by
# \latex
#   \frac{(1+\beta^2)}{\beta^2}\frac{Precision \dot Recall}{Precision + Recall}
# \latex
# \beta is to select the importance to focus on precision or recall
# if \beta > 1, precision is more important
# if \beta < 1, recall is more important
# if \beta = 1, precision and recall is both important
# the meaning of fx_score of x is the value \beta
def f_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5, beta: float=1.0) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        pred_binary = (preds == i).astype(np.float32)
        target_binary = (targets == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP
        FN = np.sum(target_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        rate = 1 / (beta**2) + 1 
        f = rate * (precision * recall + smooth) / (precision + recall + smooth) if (precision + recall) > 0 else 0.0
        scores.append(f)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def recall_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FN = np.sum(target_binary) - TP

        recall = (TP + smooth) / (TP + FN + smooth) if (TP + FN) > 0 else 0.0
        scores.append(recall)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def accuracy_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        correct = np.sum(pred_binary * target_binary)
        total = pred_binary.size
        
        accuracy = (correct + smooth) / (total + smooth) if total > 0 else 0.0
        scores.append(accuracy)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def precision_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP

        precision = (TP + smooth) / (TP + FP + smooth) if (TP + FP) > 0 else 0.0
        scores.append(precision)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def focal_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5) -> ScoreDict:
    scores = []

    for i in range(n_classes):
        target_binary = (targets == i).astype(np.float32)
        pred_binary = (preds == i).astype(np.float32)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary)

        precision = (TP + smooth) / (TP + FP + smooth) if (TP + FP) > 0 else 0.0
        scores.append(precision)

    return {
        "all": scores,
        "mean": np.mean(scores),
    }


def scores(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5, beta: float=1.0) -> AllScoreDict:
    miou = iou_score(targets, preds, n_classes, smooth, beta)
    dice = dice_score(targets, targets, n_classes, smooth, beta)
    recall = recall_score(targets, preds, n_classes, smooth)
    f1 = f1_score(targets, targets, n_classes, smooth)
    f2 = f_score(targets, targets, n_classes, smooth, 2.0)
    precision = precision_score(targets, preds, n_classes, smooth)
    accuracy = accuracy_score(targets, preds, n_classes, smooth)

    return {
        "miou": miou,
        "dice": dice,
        "f1": f1,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }
