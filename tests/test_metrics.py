import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import unittest
from utils.metrics import *

class TestMetricsFunctions(unittest.TestCase):
    def setUp(self):
        # Example test data
        self.targets = np.array([0, 1, 1, 2, 2, 2, 0, 1, 0])
        self.preds = np.array([0, 1, 0, 2, 2, 1, 0, 1, 1])
        self.n_classes = 3
        self.average = "macro"

    def test_get_metrics(self):
        metrics = get_metrics(self.targets, self.preds, self.n_classes, self.average)

        # Compute expected values manually or using sklearn
        expected_recall = recall_score(self.targets, self.preds, average=self.average)
        expected_precision = precision_score(
            self.targets, self.preds, average=self.average
        )
        expected_f1 = f1_score(self.targets, self.preds, average=self.average)

        # You can use the actual implementations of mPA and mIoU or mock them
        expected_mPA = mPA(self.targets, self.preds, self.n_classes)
        expected_mIoU = mIoU(self.targets, self.preds, self.n_classes)

        self.assertAlmostEqual(metrics["mPA"], expected_mPA, places=2)
        self.assertAlmostEqual(metrics["mIoU"], expected_mIoU, places=2)
        self.assertAlmostEqual(metrics["recall"], expected_recall, places=2)
        self.assertAlmostEqual(metrics["precision"], expected_precision, places=2)
        self.assertAlmostEqual(metrics["f1"], expected_f1, places=2)

    def test_calculate_average_metrics(self):
        metrics_list = [
            {"mPA": 0.5, "mIoU": 0.6, "recall": 0.7, "precision": 0.8, "f1": 0.75},
            {"mPA": 0.55, "mIoU": 0.65, "recall": 0.75, "precision": 0.85, "f1": 0.80},
            {"mPA": 0.60, "mIoU": 0.70, "recall": 0.80, "precision": 0.90, "f1": 0.85},
        ]

        avg_metrics = calculate_average_metrics(metrics_list)

        expected_avg_metrics = {
            "mPA": (0.5 + 0.55 + 0.60) / 3,
            "mIoU": (0.6 + 0.65 + 0.70) / 3,
            "recall": (0.7 + 0.75 + 0.80) / 3,
            "precision": (0.8 + 0.85 + 0.90) / 3,
            "f1": (0.75 + 0.80 + 0.85) / 3,
        }

        for key in expected_avg_metrics:
            self.assertAlmostEqual(
                avg_metrics[key], expected_avg_metrics[key], places=2
            )
