import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from utils.visualization import *

class TestShowImages(unittest.TestCase):

    def setUp(self):
        # 创建一些测试图像
        self.images = [Image.fromarray(np.random.rand(100, 100, 3) * 255).convert('RGB') for _ in range(6)]

    def test_show_images_basic(self):
        try:
            show_images(self.images, 2, 3)
        except Exception as e:
            self.fail(f"show_images raised an exception: {e}")

    def test_show_images_with_titles(self):
        titles = [f"Image {i+1}" for i in range(6)]
        try:
            show_images(self.images, 2, 3, titles=titles)
        except Exception as e:
            self.fail(f"show_images with titles raised an exception: {e}")

    def test_show_images_with_scale(self):
        try:
            show_images(self.images, 2, 3, scale=2.0)
        except Exception as e:
            self.fail(f"show_images with scale raised an exception: {e}")

    def test_show_images_invalid_nrows_ncols(self):
        with self.assertRaises(AssertionError):
            show_images(self.images, 0, 3)

class TestShowImageComparison(unittest.TestCase):

    def setUp(self):
        # 创建测试图像
        self.pre = Image.fromarray(np.random.rand(100, 100, 3) * 255).convert('RGB')
        self.post = Image.fromarray(np.random.rand(100, 100, 3) * 255).convert('RGB')
        self.mask = Image.fromarray(np.random.rand(100, 100) * 255).convert('L')

    def test_show_image_comparison_basic(self):
        try:
            show_image_comparison(self.pre, self.post, self.mask)
        except Exception as e:
            self.fail(f"show_image_comparison raised an exception: {e}")

    def test_show_image_comparison_with_titles(self):
        titles = ["Pre", "Post", "Mask"]
        try:
            show_image_comparison(self.pre, self.post, self.mask, titles=titles)
        except Exception as e:
            self.fail(f"show_image_comparison with titles raised an exception: {e}")

    def test_show_image_comparison_with_filename(self):
        try:
            show_image_comparison(self.pre, self.post, self.mask, filename='test_comparison.png')
        except Exception as e:
            self.fail(f"show_image_comparison with filename raised an exception: {e}")

    def test_show_image_comparison_invalid_titles(self):
        with self.assertRaises(AssertionError):
            show_image_comparison(self.pre, self.post, self.mask, titles=["Pre", "Post"])


class TestDrawMetrics(unittest.TestCase):

    def setUp(self):
        self.metrics = {
            'accuracy': 0.9,
            'precision': 0.8,
            'recall': 0.7
        }
        self.colors = ['r', 'g', 'b']

    def test_draw_metrics_basic(self):
        try:
            draw_metrics(self.metrics, self.colors, title="Metrics")
        except Exception as e:
            self.fail(f"draw_metrics raised an exception: {e}")

    def test_draw_metrics_with_filename(self):
        try:
            draw_metrics_graph(self.metrics, self.colors, title="Metrics", filename='metrics.png')
        except Exception as e:
            self.fail(f"draw_metrics with filename raised an exception: {e}")

    def test_draw_metrics_invalid_selected(self):
        with self.assertRaises(AssertionError):
            draw_metrics(self.metrics, self.colors, selected=[])


class TestDrawLossGraph(unittest.TestCase):

    def setUp(self):
        self.losses = [0.5, 0.4, 0.3, 0.2, 0.1]

    def test_draw_loss_graph_basic(self):
        try:
            draw_loss_graph(self.losses, title="Loss Graph")
        except Exception as e:
            self.fail(f"draw_loss_graph raised an exception: {e}")

    def test_draw_loss_graph_with_filename(self):
        try:
            draw_loss_graph(self.losses, title="Loss Graph", filename='loss_graph.png')
        except Exception as e:
            self.fail(f"draw_loss_graph with filename raised an exception: {e}")

    def test_draw_loss_graph_invalid_losses(self):
        with self.assertRaises(AssertionError):
            draw_loss_graph([])
