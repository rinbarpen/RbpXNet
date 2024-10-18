import matplotlib.pyplot as plt
from typing import Optional


def draw_images(
    images,
    nrows_cols: tuple[int, int],
    scale: float = 1.0,
    titles: Optional[str] = None,
    filename: Optional[str] = None,
):
    nrows, ncols = nrows_cols
    figsize = (nrows * scale, ncols * scale)
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i, j]
            image = images[i * ncols + j]
            ax.imshow(image)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i * ncols + j])

    plt.savefig(filename) if filename else plt.show()
