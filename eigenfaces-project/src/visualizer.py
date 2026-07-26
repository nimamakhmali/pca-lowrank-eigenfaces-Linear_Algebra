
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
        
        

# بخش ۳ — مرکزسازی و کوواریانس
# (اضافه کردن به visualizer.py)

def plot_mean_face(
    mean: np.ndarray,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    چهره میانگین را نمایش می‌دهد.

    Parameters
    ----------
    mean : np.ndarray, shape (4096,)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mean.reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
              cmap=CMAP_FACE, vmin=0, vmax=1)
    ax.set_title("The Average Face\n"
                 r"$\mu = \frac{1}{m}\sum_{i=1}^{m} x_i$",
                 fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def plot_deviations(
    B: np.ndarray,
    y: np.ndarray,
    n_show: int = 10,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    انحراف چند تصویر از میانگین را نمایش می‌دهد.

    مقادیر مثبت: روشن‌تر از میانگین
    مقادیر منفی: تاریک‌تر از میانگین

    Parameters
    ----------
    B      : np.ndarray, shape (m, 4096)  — داده مرکزشده
    n_show : int  — تعداد تصاویر نمایش داده‌شده
    """
    n_show = min(n_show, len(B))
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.5))

    # بیشینه قدرمطلق برای نرمال‌سازی یکسان
    vmax = np.max(np.abs(B[:n_show]))

    for i, ax in enumerate(axes):
        deviation = B[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        im = ax.imshow(deviation, cmap="RdGy",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"P{y[i]}", fontsize=9)
        ax.axis("off")

    # colorbar مشترک
    fig.colorbar(im, ax=axes.tolist(),
                 fraction=0.02, pad=0.02,
                 label="Deviation from Mean")

    fig.suptitle(
        r"$b_i = x_i - \mu$  (Deviation from Mean Face)",
        fontsize=13, y=1.02
    )
    _save_and_show(fig, save_path, show)
    return fig


def plot_centering_effect(
    X: np.ndarray,
    B: np.ndarray,
    mean: np.ndarray,
    n_show: int = 5,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    مقایسه تصاویر اصلی، میانگین و انحراف را نشان می‌دهد.

    سطر ۱: تصاویر اصلی X
    سطر ۲: چهره میانگین (یکسان برای همه)
    سطر ۳: انحراف B = X - mean
    """
    n_show = min(n_show, len(X))
    fig, axes = plt.subplots(3, n_show,
                              figsize=(n_show * 2.5, 8))

    vmax_dev = np.max(np.abs(B[:n_show]))

    for col in range(n_show):
        # سطر ۱: تصویر اصلی
        axes[0, col].imshow(
            X[col].reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
            cmap=CMAP_FACE, vmin=0, vmax=1
        )
        axes[0, col].set_title(f"Image {col+1}", fontsize=9)

        # سطر ۲: میانگین
        axes[1, col].imshow(
            mean.reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
            cmap=CMAP_FACE, vmin=0, vmax=1
        )
        if col == 0:
            axes[1, col].set_ylabel("Mean", fontsize=10,
                                     rotation=0, labelpad=40)

        # سطر ۳: انحراف
        im = axes[2, col].imshow(
            B[col].reshape(IMAGE_HEIGHT, IMAGE_WIDTH),
            cmap="RdGy",
            vmin=-vmax_dev, vmax=vmax_dev
        )
        if col == 0:
            axes[2, col].set_ylabel("Deviation", fontsize=10,
                                     rotation=0, labelpad=45)

    for ax in axes.flat:
        ax.axis("off")

    row_labels = ["Original  $x_i$",
                  r"Mean  $\mu$",
                  r"Centered  $b_i = x_i - \mu$"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10,
                                 rotation=0,
                                 labelpad=60,
                                 va="center")
        axes[row, 0].axis("on")
        axes[row, 0].tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False
        )
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle("Effect of Centering: $X \\to B = X - \\mu$",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def plot_covariance_heatmap(
    C: np.ndarray,
    n_features: int = 100,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    تصویر گرمایی از بخشی از ماتریس کوواریانس.

    نمایش کل ماتریس 4096×4096 امکان‌پذیر نیست.
    به‌جای آن، زیرماتریس n_features × n_features اول را نشان می‌دهیم.

    Parameters
    ----------
    C          : np.ndarray, shape (n, n)
    n_features : int  — تعداد ویژگی‌های نمایش داده‌شده
    """
    C_sub = C[:n_features, :n_features]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── هیت‌مپ کوواریانس ──────────────────────────
    vmax = np.max(np.abs(C_sub))
    im1  = axes[0].imshow(C_sub, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax,
                           aspect="auto")
    axes[0].set_title(
        f"Covariance Matrix C\n(first {n_features}×{n_features})",
        fontsize=12
    )
    axes[0].set_xlabel("Feature index", fontsize=10)
    axes[0].set_ylabel("Feature index", fontsize=10)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # ── واریانس هر ویژگی (قطر اصلی) ──────────────
    variances = np.diag(C)
    axes[1].plot(variances, color="steelblue",
                  linewidth=0.8, alpha=0.9)
    axes[1].fill_between(range(len(variances)),
                          variances,
                          alpha=0.3, color="steelblue")
    axes[1].set_xlabel("Feature Index (Pixel Position)", fontsize=10)
    axes[1].set_ylabel("Variance", fontsize=10)
    axes[1].set_title(
        "Per-Feature Variance: diag(C)\n"
        r"$C_{ii} = \text{Var}(\text{feature}_i)$",
        fontsize=12
    )
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Covariance Matrix Analysis\n"
        f"C ∈ ℝ^{{{C.shape[0]}×{C.shape[1]}}}, "
        f"Tr(C) = {np.trace(C):.2f}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def plot_psd_verification(
    B: np.ndarray,
    C: np.ndarray,
    n_trials: int = 1000,
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    مثبت نیمه‌معین بودن C را به‌صورت تجربی نمایش می‌دهد.

    برای n_trials بردار تصادفی v:
        vᵀCv = ||Bv||² / (m-1) ≥ 0

    نمودار توزیع vᵀCv را نشان می‌دهد.
    """
    np.random.seed(42)
    n  = C.shape[0]
    m  = B.shape[0]

    # بردارهای تصادفی نرمال‌شده
    V           = np.random.randn(n, n_trials)
    V           = V / np.linalg.norm(V, axis=0)    # نرمال‌سازی

    # محاسبه vᵀCv از دو روش
    # روش ۱: مستقیم
    vtCv_direct = np.einsum('ij,ij->j', V, C @ V)

    # روش ۲: از طریق B (باید یکسان باشد)
    vtCv_via_B  = np.sum((B @ V) ** 2, axis=0) / (m - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── توزیع vᵀCv 
    axes[0].hist(vtCv_direct, bins=50,
                  color="steelblue", alpha=0.8,
                  edgecolor="none")
    axes[0].axvline(0, color="crimson", linewidth=2,
                     linestyle="--", label="0 (PSD boundary)")
    axes[0].set_xlabel(r"$v^T C v$", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        r"Distribution of $v^T C v$ for Random Unit Vectors"
        f"\n(n_trials = {n_trials})",
        fontsize=12
    )
    axes[0].legend()
    min_val = vtCv_direct.min()
    axes[0].text(0.05, 0.85,
                  f"min(vᵀCv) = {min_val:.2e}",
                  transform=axes[0].transAxes,
                  fontsize=10,
                  color="green" if min_val >= 0 else "red")

    # ── مقایسه دو روش محاسبه 
    axes[1].scatter(vtCv_direct[:200],
                     vtCv_via_B[:200],
                     alpha=0.5, s=10,
                     color="darkorange")
    lim = max(vtCv_direct[:200].max(),
              vtCv_via_B[:200].max())
    axes[1].plot([0, lim], [0, lim],
                  "k--", linewidth=1, label="y = x")
    axes[1].set_xlabel(r"$v^T C v$ (direct)", fontsize=11)
    axes[1].set_ylabel(r"$\|Bv\|^2 / (m-1)$", fontsize=11)
    axes[1].set_title(
        r"Verification: $v^T C v = \|Bv\|^2 / (m-1)$",
        fontsize=12
    )
    axes[1].legend()

    fig.suptitle(
        "Positive Semi-Definiteness of C\n"
        r"$\forall v: v^T C v = \frac{1}{m-1}\|Bv\|^2 \geq 0$",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig        