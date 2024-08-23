import numpy as np
import torch

from sklearn import metrics
from typing import List, Dict

# metric_name_list = ['mIoU', 'mPa', 'recall', 'precision', 'f1']
metric_name_list = 'mIoU accuracy recall precision f1'.split()


def precision_score(targets, preds, average, in_build=True):
  targets = targets.reshape(-1)
  preds = preds.reshape(-1)
  return metrics.precision_score(targets, preds, average=average)


def recall_score(targets, preds, average, in_build=True):
  targets = targets.reshape(-1)
  preds = preds.reshape(-1)
  return metrics.recall_score(targets, preds, average=average)


def f1_score(targets, preds, average, in_build=True):
  targets = targets.reshape(-1)
  preds = preds.reshape(-1)
  return metrics.f1_score(targets, preds, average=average)


def mIoU(targets, preds, n_classes: int):
  assert n_classes > 0, 'The number of classes should be greater than 0'

  IoUs = []
  for c in range(n_classes):
    pred = (preds == c)
    target = (targets == c)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    IoU = intersection / union if union > 0 else 0.0
    IoUs.append(IoU)

  return np.mean(IoUs)


def accuracy_score(targets, preds, n_classes: int):
  assert n_classes > 0, 'The number of classes should be greater than 0'

  accuracies = []
  for c in range(n_classes):
    pred = (preds == c)
    target = (targets == c)
    
    correct = np.logical_and(pred, target).sum()
    total = target.sum()
    
    accuracy = correct / total if total > 0 else 0
    accuracies.append(accuracy)
  
  return np.mean(accuracies)


def get_metrics(targets, preds, n_classes: int, average: str, selected: List[str]=['accuracy', 'mIoU', 'recall', 'precision', 'f1']):
  assert n_classes > 0, 'The number of classes should be greater than 0'
  
  results = dict()
  if 'accuracy' in selected:
    results['accuracy'] = accuracy_score(targets, preds, n_classes)  
  if 'mIoU' in selected:
    results['mIoU'] = mIoU(targets, preds, n_classes)
  if 'recall' in selected:
    results['recall'] = recall_score(targets, preds, average=average)
  if 'precision' in selected:
    results['precision'] = precision_score(targets, preds, average=average)
  if 'f1' in selected:
    results['f1'] = f1_score(targets, preds, average=average)
  
  return results


def calculate_average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
  """计算每个指标的平均值"""
  # 初始化一个字典来存储每个指标的总和
  metrics_sum = {key: 0.0 for key in metrics_list[0].keys()}

  # 统计指标总和
  for metrics in metrics_list:
    for key, value in metrics.items():
      metrics_sum[key] += value
  
  # 计算平均值
  n = len(metrics_list)
  metrics_avg = {key: value / n for key, value in metrics_sum.items()}
  
  return metrics_avg
