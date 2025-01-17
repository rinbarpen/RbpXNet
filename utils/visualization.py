from typing import List, Union, Optional, Dict, Literal

import numpy as np
import seaborn as sns
import torch
import umap
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import create_file_parents
from utils.metrics.scores import fpr_score, tpr_score

# ImageType = Union[torch.Tensor, np.ndarray, Image.Image]
# ArrayLike = Union[List[float], np.ndarray, torch.Tensor]


def save_graph(filename: str) -> None:
    create_file_parents(filename)
    plt.savefig(filename)
    plt.close()


def save_or_show(filename: Optional[str]):
    if filename:
        save_graph(filename)
    else:
        plt.show()


# def show_images(images: List[ImageType],
#                 nrows: int, ncols: int,
#                 scale: float = 1.0,
#                 titles: Optional[List[str]] = None):
#     assert nrows > 0 and ncols > 0, 'nrows and ncols should be greater than 0'

#     figsize = (ncols * scale, nrows * scale)
#     _, axes = plt.subplots(nrows, ncols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, images)):
#         if torch.is_tensor(img):
#             img = img.numpy()
#         ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     plt.show()


# def show_image_comparison(pre: ImageType, post: ImageType, mask: ImageType,
#                           titles: Optional[List[str]] = None,
#                           filename: Optional[str] = None):
#     assert titles is None or len(titles) == 3, 'Images must have the same number of titles'

#     figsize = (10, 10)
#     _, axes = plt.subplots(2, 2, figsize=figsize)

#     axes[0, 0].axis('off')
#     axes[0, 1].axis('off')
#     axes[1, 0].axis('off')
#     axes[1, 1].axis('off')

#     axes[0, 0].imshow(pre)
#     axes[0, 1].imshow(post)
#     axes[1, 0].imshow(mask)
#     axes[1, 1].set_position([0.25, 0.1, 0.5, 0.5])

#     if titles:
#         axes[0, 0].set_title(titles[0])
#         axes[0, 1].set_title(titles[1])
#         axes[1, 0].set_title(titles[2])

#     plt.subplots_adjust(wspace=0.1, hspace=0.3)

#     save_or_show(filename)


def draw_loss_graph(
    losses: List[float],
    step: int,
    filename: Optional[str] = None,
    title: Optional[str] = None,
):
    plt.figure()
    plt.plot([(i + 1) for i in range(0, len(losses), step)], losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title:
        plt.title(title)

    save_or_show(filename)


def draw_heat_graph(
    possibility_matrix: Union[np.ndarray, Image.Image, torch.Tensor],
    filename: Optional[str] = None,
    title: Optional[str] = None,
    x_ticks=False,
    y_ticks=False,
    x_label: str = "",
    y_label: str = "",
):
    """
    Draws a heatmap from a 2D array representing a probability distribution.

    Parameters:
    - possibility_matrix (np.ndarray): A 2D numpy array representing the probability distribution.
      The shape should be (H, W) and the values should be in the range [0, 1].
    - filename (str): The name of the file where the heatmap will be saved.
    - title (Optional[str]): The title of the heatmap. Default is None.
    - x_ticks (bool): Whether to display x-axis ticks. Default is False.
    - y_ticks (bool): Whether to display y-axis ticks. Default is False.
    - x_label (str): The label for the x-axis. Default is an empty string.
    - y_label (str): The label for the y-axis. Default is an empty string.

    Returns:
    None. The function saves the heatmap as an image file.
    """
    if isinstance(possibility_matrix, Image.Image):
        possibility_matrix_np = np.array(possibility_matrix)
        # assert possibility_matrix.ndim == 2, "possibility_matrix should be a 2-dimensional array"
    elif isinstance(possibility_matrix, torch.Tensor):
        possibility_matrix_np = possibility_matrix.cpu().detach().numpy()
    else:
        possibility_matrix_np = possibility_matrix

    ax = sns.heatmap(possibility_matrix_np, xticklabels=x_ticks, yticklabels=y_ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    save_or_show(filename)


def draw_attention_heat_graph(
    possibility_matrix: Union[np.ndarray, Image.Image, torch.Tensor],
    original_image: Union[np.ndarray, Image.Image, torch.Tensor],
    filename: Optional[str] = None,
    title: Optional[str] = None,
    x_ticks=False,
    y_ticks=False,
    x_label: str = "",
    y_label: str = "",
):
    """
    Draws an attention heat graph on top of an original image.

    Parameters:
    - possibility_matrix (np.ndarray): A 2D array representing the attention map.
      The shape should be (H, W) and the value range should be [0, 1].
    - original_image (Union[np.ndarray, Image.Image, torch.Tensor]): The original image on which the attention map will be drawn.
      The shape should be (H, W, C) and the value range is not limited.
    - filename (str): The name of the file where the graph will be saved.
    - title (Optional[str]): The title of the graph. Default is None.
    - x_ticks (bool): Whether to display x-axis ticks. Default is False.
    - y_ticks (bool): Whether to display y-axis ticks. Default is False.
    - x_label (str): The label for the x-axis. Default is an empty string.
    - y_label (str): The label for the y-axis. Default is an empty string.

    Returns:
    None. The function saves the attention heat graph as an image file.
    """
    if isinstance(possibility_matrix, Image.Image):
        possibility_matrix_np = np.array(possibility_matrix)
    elif isinstance(possibility_matrix, torch.Tensor):
        possibility_matrix_np = possibility_matrix.cpu().detach().numpy()
    else:
        possibility_matrix_np = possibility_matrix

    if isinstance(original_image, Image.Image):
        original_image_np = np.array(original_image)
    elif isinstance(original_image, torch.Tensor):
        original_image_np = original_image.cpu().detach().numpy()
    else:
        original_image_np = original_image

    if original_image_np.ndim == 3:
        assert (
            possibility_matrix_np.shape == original_image_np.shape[:2]
        ), "Both of them should be matched at the pixel level."
    elif original_image_np.ndim == 2:
        assert (
            possibility_matrix_np.shape == original_image_np.shape
        ), "Both of them should be matched at the pixel level."

    plt.figure()
    plt.imshow(original_image_np)
    plt.imshow(possibility_matrix_np, cmap="rainbow", alpha=0.4)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not x_ticks:
        plt.xticks([])
    if not y_ticks:
        plt.yticks([])
    if title:
        plt.title(title)

    plt.tight_layout()
    save_or_show(filename)


def draw_metrics_graph(
    metrics: Dict[str, float],
    colors: List[str],
    filename: Optional[str],
    selected: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    if selected is None:
        selected = list(metrics.keys())

    metrics_show = dict(filter(lambda item: item[0] in selected, metrics.items()))

    bar_width = min(1 / len(metrics), 0.2)

    plt.figure()
    plt.bar(
        list(metrics_show.keys()),
        list(metrics_show.values()),
        color=colors,
        width=bar_width,
    )

    plt.xlabel("Metrics")
    plt.ylabel("Values")

    plt.ylim([0, 1])

    if title:
        plt.title(title)

    save_or_show(filename)


def draw_roc_curve(
    metrics: dict[str, list[dict[str, float]]],
    colors: tuple[str],
    title: Optional[str] = None,
    filename: Optional[str] = None,
):
    tprs = dict()  # label: list of tpr
    fprs = dict()  # label: list of fpr
    for metric in metrics["tpr"]:
        for k, v in metric.items():
            tprs[k].append(v)
    for metric in metrics["fpr"]:
        for k, v in metric.items():
            fprs[k].append(v)

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if title:
        plt.title(title)

    save_or_show(filename)

def draw_auc_curve(
    metrics: dict[str, list[dict[str, float]]],
    colors: tuple[str],
    title: Optional[str] = None,
    filename: Optional[str] = None,
):
    tprs = dict()  # label: list of tpr
    fprs = dict()  # label: list of fpr
    for metric in metrics["tpr"]:
        for k, v in metric.items():
            tprs[k].append(v)
    for metric in metrics["fpr"]:
        for k, v in metric.items():
            fprs[k].append(v)

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if title:
        plt.title(title)

    save_or_show(filename)


def high_dimension_vision(
    x: np.ndarray,
    method: Literal["umap", "pca", "tsne"],
    xlabel: str,
    ylabel: str,
    filename: Optional[str] = None,
    **kwargs,
):
    match method:
        case "pca":
            pca = PCA(n_components=kwargs["n_components"])
            x_pca = pca.fit_transform(x)

            plt.scatter(x_pca[:, 0], x_pca[:, 1])
            plt.title("PCA")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        case "tsne":
            tsne = TSNE(
                n_components=kwargs["n_components"], random_state=kwargs["random_state"]
            )
            x_tsne = tsne.fit_transform(x)

            plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
            plt.title("t-SNE")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        case "umap":
            umap_model = umap.UMAP(n_components=2, random_state=0)
            x_umap = umap_model.fit_transform(x)

            plt.scatter(x_umap[:, 0], x_umap[:, 1])
            plt.title("UMAP")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    save_or_show(filename)
