import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import unittest
from utils.metrics import *

class TestMetricsFunctions(unittest.TestCase):
    def setUp(self):
        # Example test data
        self.targets = torch.randn((2, 1, 512, 512)) * 256
        self.preds = torch.randn((2, 1, 512, 512)) * 256
        self.targets = self.targets.type(dtype=torch.int8).numpy()
        self.preds = self.preds.type(dtype=torch.int8).numpy()
        self.n_classes = 1
        self.average = "macro"

    def test_mIoU(self):
        mIoU_value = mIoU(self.targets, self.preds, self.n_classes)
        print(f"{mIoU_value=}")
        self.assertTrue(0 <= mIoU_value <= 1)
    
    def test_accuracy_score(self):
        accuracy_value = accuracy_score(self.targets, self.preds, self.n_classes)
        print(f"{accuracy_value=}")
        self.assertTrue(0 <= accuracy_value <= 1)

    def test_precision_score(self):
        precision_value = precision_score(self.targets, self.preds, self.average)
        print(f"{precision_value=}")
        self.assertTrue(0 <= precision_value <= 1)

    def test_recall_score(self):
        recall_value = recall_score(self.targets, self.preds, self.average)
        print(f"{recall_value=}")
        self.assertTrue(0 <= recall_value <= 1)
    
    def test_f1_score(self):
        f1_value = f1_score(self.targets, self.preds, self.average)
        print('\n'
          .join(f"; {repr(f1_value)=}; {isinstance(f1_value, float)=}; {(0<=f1_value<=1)=}"
          .split(';'))) 
        self.assertTrue(0 <= f1_value <= 1)
    
    def test_dice_score(self):
        # dice_value = dice_score(self.targets, self.preds)
        # print(f"{dice_value=}")
        pass

    def test_get_metrics(self):
        metrics = get_metrics(self.targets, self.preds, self.n_classes, self.average)

        # Compute expected values manually or using sklearn
        expected_recall = recall_score(self.targets, self.preds, average=self.average)
        expected_precision = precision_score(
            self.targets, self.preds, average=self.average
        )
        expected_f1 = f1_score(self.targets, self.preds, average=self.average)

        # You can use the actual implementations of mPA and mIoU or mock them
        expected_accuracy = accuracy_score(self.targets, self.preds, self.n_classes)
        expected_mIoU = mIoU(self.targets, self.preds, self.n_classes)

        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy, places=2)
        self.assertAlmostEqual(metrics["mIoU"], expected_mIoU, places=2)
        self.assertAlmostEqual(metrics["recall"], expected_recall, places=2)
        self.assertAlmostEqual(metrics["precision"], expected_precision, places=2)
        self.assertAlmostEqual(metrics["f1"], expected_f1, places=2)

    def test_calculate_average_metrics(self):
        metrics_list = [
            {"accuracy": 0.5, "mIoU": 0.6, "recall": 0.7, "precision": 0.8, "f1": 0.75},
            {"accuracy": 0.55, "mIoU": 0.65, "recall": 0.75, "precision": 0.85, "f1": 0.80},
            {"accuracy": 0.60, "mIoU": 0.70, "recall": 0.80, "precision": 0.90, "f1": 0.85},
        ]

        avg_metrics = calculate_average_metrics(metrics_list)

        expected_avg_metrics = {
            "accuracy": (0.5 + 0.55 + 0.60) / 3,
            "mIoU": (0.6 + 0.65 + 0.70) / 3,
            "recall": (0.7 + 0.75 + 0.80) / 3,
            "precision": (0.8 + 0.85 + 0.90) / 3,
            "f1": (0.75 + 0.80 + 0.85) / 3,
        }

        for key in expected_avg_metrics:
            self.assertAlmostEqual(
                avg_metrics[key], expected_avg_metrics[key], places=2
            )
