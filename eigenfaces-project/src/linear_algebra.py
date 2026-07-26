

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EPSILON



# بخش ۱ — مرکزسازی


def center_data(X: np.ndarray) -> tuple:

    mean = np.mean(X, axis=0)   # میانگین هر ستون → shape: (n,)
    B    = X - mean              # broadcast: (m,n) - (n,) → (m,n)
    return B, mean


def verify_centering(B: np.ndarray,
                     tol: float = EPSILON) -> dict:

    column_means = np.mean(B, axis=0)         # shape: (n,)
    max_abs      = float(np.max(np.abs(column_means)))
    mean_of_means = float(np.mean(np.abs(column_means)))

    return {
        "is_centered"  : bool(max_abs < tol),
        "max_mean_abs" : max_abs,
        "mean_of_means": mean_of_means
    }



# بخش ۲ — ماتریس کوواریانس


def covariance_matrix(B: np.ndarray) -> np.ndarray:

    m = B.shape[0]
    C = (1.0 / (m - 1)) * (B.T @ B)   # (n,m)@(m,n) → (n,n)
    return C


def verify_covariance_properties(C: np.ndarray,
                                  tol: float = EPSILON) -> dict:

    #  بررسی تقارن 
    sym_error    = np.max(np.abs(C - C.T))
    is_symmetric = bool(sym_error < tol)

    #  بررسی مثبت نیمه‌معین 
    # برای سرعت، روی زیرماتریس کوچک بررسی می‌کنیم
    sample_size = min(200, C.shape[0])
    C_sample    = C[:sample_size, :sample_size]
    eigs        = np.linalg.eigvalsh(C_sample)

    min_eig        = float(eigs.min())
    n_negative     = int(np.sum(eigs < -tol))
    is_psd         = bool(min_eig >= -tol)

    return {
        "is_symmetric"  : is_symmetric,
        "is_psd"        : is_psd,
        "max_sym_error" : float(sym_error),
        "min_eigenvalue": min_eig,
        "n_negative_eigs": n_negative
    }


def covariance_diagonal(C: np.ndarray) -> np.ndarray:
 
    return np.diag(C)


def total_variance(C: np.ndarray) -> float:

    return float(np.trace(C))



# بخش ۳ — تجزیه طیفی


def compute_eigh(C: np.ndarray) -> tuple:

    # eigh خروجی صعودی می‌دهد
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # مرتب‌سازی نزولی
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors



# بخش ۴ — SVD


def compute_svd(B: np.ndarray) -> tuple:

    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return U, S, Vt


def eigenvalues_from_svd(S: np.ndarray, m: int) -> np.ndarray:

    return (S ** 2) / (m - 1)



# بخش ۵ — پروجکشن و بازسازی


def project(B: np.ndarray,
            components: np.ndarray,
            k: int) -> np.ndarray:
  
    assert k <= components.shape[1], \
        f"k={k} larger than available components={components.shape[1]}"

    W = components[:, :k]   # shape: (n, k)
    T = B @ W               # shape: (m, k)
    return T


def reconstruct(T: np.ndarray,
                components: np.ndarray,
                k: int,
                mean: np.ndarray) -> np.ndarray:
  
    W       = components[:, :k]    # shape: (n, k)
    B_recon = T @ W.T              # shape: (m, n)
    X_recon = B_recon + mean       # shape: (m, n)
    return X_recon


def lowrank_approximation_svd(B: np.ndarray,
                               U: np.ndarray,
                               S: np.ndarray,
                               Vt: np.ndarray,
                               k: int) -> np.ndarray:

    Uk  = U[:, :k]           # shape: (m, k)
    Sk  = S[:k]              # shape: (k,)
    Vtk = Vt[:k, :]          # shape: (k, n)
    B_k = Uk * Sk @ Vtk      # shape: (m, n)
    return B_k



# بخش ۶ — معیارهای خطا


def frobenius_error(A: np.ndarray, B: np.ndarray) -> float:

    return float(np.linalg.norm(A - B, 'fro'))


def relative_frobenius_error(A: np.ndarray,
                              B_approx: np.ndarray) -> float:

    norm_A = np.linalg.norm(A, 'fro')
    if norm_A < EPSILON:
        return 0.0
    return frobenius_error(A, B_approx) / norm_A


def reconstruction_mse(X_true: np.ndarray,
                        X_recon: np.ndarray) -> float:
 
    return float(np.mean((X_true - X_recon) ** 2))


def explained_energy_from_svd(S: np.ndarray, k: int) -> float:

    total_energy = np.sum(S ** 2)
    k_energy     = np.sum(S[:k] ** 2)
    return float(k_energy / total_energy)



# بخش ۷ — تحلیل رتبه


def compute_rank(A: np.ndarray, tol: float = None) -> int:

    _, S, _ = np.linalg.svd(A, full_matrices=False)

    if tol is None:
        # آستانه استاندارد MATLAB-style
        tol = max(A.shape) * np.finfo(float).eps * S[0]

    return int(np.sum(S > tol))


def rank_analysis(B: np.ndarray) -> dict:

    m, n              = B.shape
    theoretical_max   = min(m, n)
    numerical_rank    = compute_rank(B)
    null_dim          = n - numerical_rank

    return {
        "m"                  : m,
        "n"                  : n,
        "theoretical_max_rank": theoretical_max,
        "numerical_rank"     : numerical_rank,
        "null_space_dim"     : null_dim,
        "rank_fraction"      : numerical_rank / n
    }



# بخش ۸ — تحلیل طیف


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:

    total = np.sum(eigenvalues)
    if total < EPSILON:
        return np.zeros_like(eigenvalues)
    return eigenvalues / total


def cumulative_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:

    return np.cumsum(explained_variance_ratio(eigenvalues))


def n_components_for_threshold(eigenvalues: np.ndarray,
                                threshold: float) -> int:

    cvr = cumulative_variance_ratio(eigenvalues)
    idx = np.argmax(cvr >= threshold)
    return int(idx) + 1