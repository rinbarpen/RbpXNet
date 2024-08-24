import unittest

from utils.metrics.metrics import *


class TestMetricsFunctions(unittest.TestCase):
    def setUp(self):
        # Example test data
        # self.targets = torch.randn(2, 1, 512, 512).numpy()
        # self.preds = torch.randn(2, 1, 512, 512).numpy()
        self.targets = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
                            [[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]]]])
        self.preds = np.array(
            [[[[0.1, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]], [[0.2, 0.6, 0.2], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]],
             [[[0.1, 0.1, 0.1], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]],
              [[0.1, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.6, 0.3]]]])
        self.preds[self.preds >= 0.5] = 1
        self.preds[self.preds <= 0.5] = 0

        self.labels = ["X", "Y"]
        self.average = "macro"

    def test_mIoU(self):
        ious, miou, iou_mapper = iou_score(self.targets, self.preds, self.labels)
        print(f"{repr(ious)}, {repr(miou)}, {repr(iou_mapper)}")
        self.assertTrue(0 <= miou <= 1)
    
    def test_accuracy_score(self):
        accuracy_value = accuracy_score(self.targets, self.preds, self.labels)
        print(f"{accuracy_value=}")
        self.assertTrue(0 <= accuracy_value <= 1)

    def test_precision_score(self):
        precision_value = precision_score(self.targets, self.preds, self.labels)
        print(f"{precision_value=}")

    def test_recall_score(self):
        recall_value = recall_score(self.targets, self.preds, self.labels)
        print(f"{recall_value=}")

    def test_f1_score(self):
        f1_value = f1_score(self.targets, self.preds, self.labels)
        print(f"{f1_value=}")

    def test_dice_score(self):
        dice_value = dice_score(self.targets, self.preds, self.labels)
        print(f"{dice_value=}")

    def test_pa_score(self):
        pa_value = pa_score(self.targets, self.preds, self.labels)
        print(f"{pa_value=}")

    def test_roc_auc_score(self):
        roc_value = roc_auc_score(self.targets, self.preds, self.labels)
        auc_value = auc(self.targets, self.preds, self.labels)
        print(f"{roc_value=}, {auc_value=}")
