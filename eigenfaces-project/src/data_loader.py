# src/data_loader.py
# ═══════════════════════════════════════════════════════════════
# مسئولیت: بارگذاری، اعتبارسنجی، و تقسیم داده‌های Olivetti
#
# این ماژول تنها نقطه تماس با داده خام است.
# هیچ ماژول دیگری مستقیماً داده بارگذاری نمی‌کند.
# ═══════════════════════════════════════════════════════════════

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import sys
import os

# اضافه کردن مسیر ریشه برای import از config
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    DATA_DIR,
    N_SAMPLES,
    N_PERSONS,
    N_FEATURES,
    N_TRAIN_PER_PERSON,
    RANDOM_STATE
)


# ══════════════════════════════════════════════════════
# توابع اصلی
# ══════════════════════════════════════════════════════

def load_faces(
    shuffle: bool = True,
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    داده‌های Olivetti Faces را بارگذاری و اعتبارسنجی می‌کند.

    این مجموعه شامل ۴۰۰ تصویر از ۴۰ نفر مختلف است.
    هر تصویر ۶۴×۶۴ پیکسل دارد و به‌صورت بردار ۴۰۹۶ بعدی ذخیره شده.
    مقادیر پیکسل در بازه [0, 1] نرمال‌سازی شده‌اند.

    Parameters
    ----------
    shuffle : bool
        آیا داده‌ها قبل از بازگشت مخلوط شوند؟
    random_state : int
        seed برای تکرارپذیری

    Returns
    -------
    X : np.ndarray, shape (400, 4096)
        ماتریس داده — هر سطر یک تصویر مسطح‌شده است
    y : np.ndarray, shape (400,)
        برچسب هر تصویر — شناسه فرد از 0 تا 39

    Examples
    --------
    >>> X, y = load_faces()
    >>> X.shape
    (400, 4096)
    >>> np.unique(y)
    array([ 0,  1,  2, ..., 39])
    """
    print("Loading Olivetti Faces dataset...")

    faces = fetch_olivetti_faces(
        data_home=DATA_DIR,
        shuffle=shuffle,
        random_state=random_state
    )

    X = faces.data      # shape: (400, 4096)
    y = faces.target    # shape: (400,)

    # اعتبارسنجی پیش از بازگشت
    _validate(X, y)

    print(f"  Loaded: {X.shape[0]} images, "
          f"{X.shape[1]} features per image")
    print(f"  Persons: {len(np.unique(y))}")
    print(f"  Pixel range: [{X.min():.3f}, {X.max():.3f}]")
    print("  Done.\n")

    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = N_TRAIN_PER_PERSON
) -> tuple:
    """
    داده را به مجموعه آموزش و آزمون تقسیم می‌کند.

    استراتژی تقسیم:
        برای هر فرد، اولین n_train تصویر → آموزش
        باقی‌مانده تصاویر       → آزمون

    این استراتژی تضمین می‌کند که هر فرد هم در آموزش
    و هم در آزمون نمایندگی دارد (stratified split).

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        ماتریس داده
    y : np.ndarray, shape (m,)
        برچسب‌ها
    n_train : int
        تعداد تصاویر هر فرد در مجموعه آموزش

    Returns
    -------
    X_train : np.ndarray, shape (n_train * N_PERSONS, n)
    y_train : np.ndarray, shape (n_train * N_PERSONS,)
    X_test  : np.ndarray, shape (n_test * N_PERSONS, n)
    y_test  : np.ndarray, shape (n_test * N_PERSONS,)
    """
    train_idx = []
    test_idx  = []

    for person_id in np.unique(y):
        # ایندکس تمام تصاویر این فرد
        person_mask   = np.where(y == person_id)[0]

        train_idx.extend(person_mask[:n_train])
        test_idx.extend(person_mask[n_train:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"Train set: {X_train.shape[0]} images "
          f"({n_train} per person)")
    print(f"Test  set: {X_test.shape[0]} images "
          f"({N_PERSONS - n_train + (N_IMAGES_PER_PERSON - n_train)} "
          f"per person)")

    return X_train, y_train, X_test, y_test


def get_person_images(
    X: np.ndarray,
    y: np.ndarray,
    person_id: int
) -> np.ndarray:
    """
    تمام تصاویر یک فرد مشخص را برمی‌گرداند.

    Parameters
    ----------
    person_id : int
        شناسه فرد (0 تا 39)

    Returns
    -------
    np.ndarray, shape (n_images, 4096)
    """
    mask = y == person_id
    return X[mask]


# ══════════════════════════════════════════════════════
# توابع کمکی داخلی
# ══════════════════════════════════════════════════════

def _validate(X: np.ndarray, y: np.ndarray) -> None:
    """
    اعتبارسنجی ساختار و محتوای داده‌ها.
    در صورت وجود مشکل، خطای توصیفی ایجاد می‌کند.
    """
    if X.shape != (N_SAMPLES, N_FEATURES):
        raise ValueError(
            f"Expected X.shape = ({N_SAMPLES}, {N_FEATURES}), "
            f"got {X.shape}"
        )

    if len(np.unique(y)) != N_PERSONS:
        raise ValueError(
            f"Expected {N_PERSONS} persons, "
            f"got {len(np.unique(y))}"
        )

    if not (X.min() >= 0 and X.max() <= 1):
        raise ValueError(
            f"Pixel values should be in [0, 1], "
            f"got [{X.min():.3f}, {X.max():.3f}]"
        )

    if X.dtype != np.float64:
        raise TypeError(
            f"Expected float64, got {X.dtype}"
        )


# ══════════════════════════════════════════════════════
# آمار توصیفی
# ══════════════════════════════════════════════════════

def dataset_summary(X: np.ndarray, y: np.ndarray) -> dict:
    """
    آمار توصیفی مجموعه داده را محاسبه و برمی‌گرداند.

    Returns
    -------
    dict شامل:
        n_samples, n_features, n_persons,
        n_images_per_person, pixel_min, pixel_max,
        pixel_mean, pixel_std, max_possible_rank
    """
    persons, counts = np.unique(y, return_counts=True)

    summary = {
        "n_samples"           : X.shape[0],
        "n_features"          : X.shape[1],
        "n_persons"           : len(persons),
        "images_per_person"   : {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean())
        },
        "pixel_stats"         : {
            "min" : float(X.min()),
            "max" : float(X.max()),
            "mean": float(X.mean()),
            "std" : float(X.std())
        },
        "max_possible_rank"   : min(X.shape[0], X.shape[1]),
        "dtype"               : str(X.dtype)
    }

    return summary


def print_summary(summary: dict) -> None:
    """آمار توصیفی را به‌صورت خوانا چاپ می‌کند."""
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Samples   : {summary['n_samples']}")
    print(f"  Features  : {summary['n_features']} "
          f"({int(summary['n_features']**0.5)}x"
          f"{int(summary['n_features']**0.5)} pixels)")
    print(f"  Persons   : {summary['n_persons']}")
    print(f"  Img/Person: {summary['images_per_person']['mean']:.0f}")
    print(f"  Pixel min : {summary['pixel_stats']['min']:.4f}")
    print(f"  Pixel max : {summary['pixel_stats']['max']:.4f}")
    print(f"  Pixel mean: {summary['pixel_stats']['mean']:.4f}")
    print(f"  Pixel std : {summary['pixel_stats']['std']:.4f}")
    print(f"  Max rank  : {summary['max_possible_rank']}")
    print(f"  Dtype     : {summary['dtype']}")
    print("=" * 50)