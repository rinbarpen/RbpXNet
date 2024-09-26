import numpy as np

def iou_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6):    
    scores = np.zeros(n_classes)

    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        iou = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
        scores[i] = iou

    return {
        "all": scores,
        "mean": np.mean(scores)
    }

def dice_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6):

    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        TP_FN_FP = np.sum(pred_binary) + np.sum(target_binary)

        dice = (2.0 * TP + smooth) / (TP_FN_FP + smooth) if TP_FN_FP > 0 else 0.0
        scores[i] = dice

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
def f1_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6) -> dict:

    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP
        FN = np.sum(target_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = 2.0 * (precision * recall + smooth) / (precision + recall + smooth) if (precision + recall) > 0 else 0.0
        scores[i] = f1

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def f_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6, beta: float=2.0) -> dict: 

    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP
        FN = np.sum(target_binary) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        rate = 1 / (beta**2) + 1 
        f = rate * (precision * recall + smooth) / (precision + recall + smooth) if (precision + recall) > 0 else 0.0
        scores[i] = f

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def recall_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6) -> dict:

    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        FN = np.sum(target_binary) - TP

        recall = (TP + smooth) / (TP + FN + smooth) if (TP + FN) > 0 else 0.0
        scores[i] = recall

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def accuracy_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6) -> dict:

    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        correct = np.sum(pred_binary * target_binary)
        total = pred_binary.size
        
        accuracy = (correct + smooth) / (total + smooth) if total > 0 else 0.0
        scores[i] = accuracy

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def precision_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6) -> dict:
    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP

        precision = (TP + smooth) / (TP + FP + smooth) if (TP + FP) > 0 else 0.0
        scores[i] = precision

    return {
        "all": scores,
        "mean": np.mean(scores),
    }

def focal_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6):
    scores = np.zeros(n_classes)
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        FP = np.sum(pred_binary) - TP

        precision = (TP + smooth) / (TP + FP + smooth) if (TP + FP) > 0 else 0.0
        scores[i] = precision

    return {
        "all": scores,
        "mean": np.mean(scores),
    }
    
def calcuate_scores(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-6) -> dict:    
    ious = iou_score(targets, preds, n_classes, smooth=smooth)
    dice = dice_score(targets, targets, n_classes, smooth=smooth)
    recall = recall_score(targets, preds, n_classes, smooth=smooth)
    f1 = f1_score(targets, targets, n_classes, smooth=smooth)
    f2 = f_score(targets, targets, n_classes, smooth=smooth, beta=2.0)
    precision = precision_score(targets, preds, n_classes, smooth=smooth)
    accuracy = accuracy_score(targets, preds, n_classes, smooth=smooth)

    return {
        "miou": ious,
        "dice": dice,
        "f1": f1,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }
