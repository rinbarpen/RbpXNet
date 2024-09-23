import numpy as np

def iou_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5, beta: float=0.0):
    scores = []
    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)

        iou = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
        scores.append(iou)

    return {
        "all": scores,
        "average": np.mean(scores)
    }

def dice_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float=1e-5, beta: float=0.0):
    dice_scores = []

    for i in range(n_classes):
        target_binary = (targets == i)
        pred_binary = (preds == i)

        TP = np.sum(pred_binary * target_binary)
        TP_FN_FP = np.sum(pred_binary) + np.sum(target_binary)

        dice = (2.0 * TP + smooth) / (TP_FN_FP + smooth) if TP_FN_FP > 0 else 0.0
        dice_scores.append(dice)

    return {
        "all": dice_scores,
        "average": np.mean(dice_scores),
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
def f1_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float) -> dict:
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
        "average": np.mean(scores),
    }

def f_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float, beta: float) -> dict: 
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
        "average": np.mean(scores),
    }

def recall_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float) -> dict:
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
        "average": np.mean(scores),
    }

def accuracy_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float) -> dict:
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
        "average": np.mean(scores),
    }

def precision_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float) -> dict:
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
        "average": np.mean(scores),
    }

def focal_score(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float, beta: float):
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
        "average": np.mean(scores),
    }
    
def scores(targets: np.ndarray, preds: np.ndarray, n_classes: int, smooth: float, beta: float) -> dict:    
    ious = iou_score(targets, preds, n_classes)
    dice = dice_score(targets, targets, n_classes)
    recall = recall_score(targets, preds, n_classes, smooth)
    f1 = f1_score(targets, targets, n_classes)
    precision = precision_score(targets, preds, n_classes, smooth)
    accuracy = accuracy_score(targets, preds, n_classes, smooth)

    return {
        "miou": ious,
        "dice": dice,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }
