
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    FIGURE_DPI, FIGURE_FORMAT,
    CMAP_FACE, FIGURES_DIR, STYLE
)

#
plt.style.use(STYLE)


# بخش ۱ — نمایش تصاویر

def plot_sample_faces(
    X: np.ndarray,
    y: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 10,
    title: str = "Sample Faces from Olivetti Dataset",
    save_path: str = None,
    show: bool = True
) -> plt.Figure:

    n_images = min(n_rows * n_cols, len(X))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2.0)
    )

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            face = X[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            ax.imshow(face, cmap=CMAP_FACE, vmin=0, vmax=1)
            ax.set_title(f"P{y[i]}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    _save_and_show(fig, save_path, show)
    return fig


def plot_single_face(
    x: np.ndarray,
    title: str = "Face",
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(x.reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
              cmap=CMAP_FACE, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def plot_pixel_distribution(
    X: np.ndarray,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # هیستوگرام مقادیر پیکسل
    axes[0].hist(X.flatten(), bins=100,
                 color="steelblue", alpha=0.8,
                 edgecolor="none")
    axes[0].set_xlabel("Pixel Value", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of All Pixel Values",
                       fontsize=13)
    axes[0].axvline(X.mean(), color="crimson",
                    linestyle="--", linewidth=2,
                    label=f"Mean = {X.mean():.3f}")
    axes[0].legend()

    # میانگین مقادیر پیکسل در هر موقعیت (تصویر گرمایی)
    mean_pixels = X.mean(axis=0).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    im = axes[1].imshow(mean_pixels, cmap="viridis")
    axes[1].set_title("Average Pixel Intensity per Position",
                       fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle("Pixel Statistics", fontsize=14,
                  fontweight="bold")
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def plot_person_grid(
    X: np.ndarray,
    y: np.ndarray,
    person_ids: list,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:

    n_persons = len(person_ids)
    n_images  = 10   # تصاویر هر فرد

    fig, axes = plt.subplots(
        n_persons, n_images,
        figsize=(n_images * 1.8, n_persons * 2.0)
    )

    for row, pid in enumerate(person_ids):
        person_imgs = X[y == pid]
        for col in range(min(n_images, len(person_imgs))):
            ax = axes[row, col] if n_persons > 1 else axes[col]
            ax.imshow(
                person_imgs[col].reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
                cmap=CMAP_FACE, vmin=0, vmax=1
            )
            if col == 0:
                ax.set_ylabel(f"Person {pid}", fontsize=9)
            ax.axis("off")

    fig.suptitle("All Images per Person (Sample)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


# بخش ۲ — آمار و اطلاعات ماتریس

def plot_matrix_info(
    X: np.ndarray,
    title: str = "Data Matrix X",
    save_path: str = None,
    show: bool = True
) -> plt.Figure:

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # ۱. هیت‌مپ ۵۰ سطر اول
    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(X[:50, :100], aspect="auto",
                      cmap="viridis")
    ax1.set_xlabel("Feature index (first 100)", fontsize=10)
    ax1.set_ylabel("Sample index (first 50)", fontsize=10)
    ax1.set_title("Heatmap of X (first 50×100)", fontsize=11)
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # ۲. توزیع نرم هر سطر
    row_norms = np.linalg.norm(X, axis=1)
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(row_norms, bins=30, color="steelblue",
              alpha=0.8, edgecolor="none")
    ax2.set_xlabel("L2 Norm of Each Image", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Distribution of Row Norms", fontsize=11)
    ax2.axvline(row_norms.mean(), color="crimson",
                linestyle="--", linewidth=2,
                label=f"Mean = {row_norms.mean():.2f}")
    ax2.legend(fontsize=9)

    # ۳. توزیع نرم هر ستون
    col_norms = np.linalg.norm(X, axis=0)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(col_norms, color="darkorange",
              linewidth=0.8, alpha=0.9)
    ax3.set_xlabel("Feature Index (Pixel Position)", fontsize=10)
    ax3.set_ylabel("L2 Norm", fontsize=10)
    ax3.set_title("Column Norms (per Pixel)", fontsize=11)

    fig.suptitle(title, fontsize=14,
                  fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig



def _save_and_show(
    fig: plt.Figure,
    save_path: str = None,
    show: bool = True
) -> None:
    if save_path is not None:
  
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=FIGURE_DPI,
            format=FIGURE_FORMAT,
            bbox_inches="tight"
        )
        print(f"  Figure saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)