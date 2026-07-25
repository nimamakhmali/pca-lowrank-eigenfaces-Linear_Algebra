# src/__init__.py
# ─────────────────────────────────────
# این فایل src را به یک پکیج Python تبدیل می‌کند
# ─────────────────────────────────────

from .data_loader    import load_faces, split_train_test
from .linear_algebra import (center_data, covariance_matrix,
                              compute_eigh, compute_svd,
                              project, reconstruct)
from .visualizer     import (plot_faces, plot_eigenfaces,
                              plot_spectrum, plot_reconstruction,
                              plot_error_curve)

__version__ = "1.0.0"
__author__  = "Your Name"