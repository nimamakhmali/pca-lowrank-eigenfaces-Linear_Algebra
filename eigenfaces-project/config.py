# config.py
# تنظیمات مرکزی پروژه Eigenfaces
# تمام مقادیر ثابت اینجا تعریف می‌شوند
# هیچ‌گاه magic number در کد اصلی استفاده نشود

import os

# مسیرها
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
REPORT_DIR  = os.path.join(ROOT_DIR, "report")

# مشخصات داده
IMAGE_HEIGHT        = 64
IMAGE_WIDTH         = 64
N_FEATURES          = IMAGE_HEIGHT * IMAGE_WIDTH   # 4096
N_SAMPLES           = 400
N_PERSONS           = 40
N_IMAGES_PER_PERSON = 10
RANDOM_STATE        = 42

# تنظیمات PCA
K_VALUES_RECONSTRUCTION = [1, 5, 10, 20, 30, 50, 100, 150, 200, 300, 400]
K_VALUES_RECOGNITION    = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
K_DEFAULT               = 50

# آستانه‌های عددی
EPSILON             = 1e-10
VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

# تقسیم داده
N_TRAIN_PER_PERSON = 8
N_TEST_PER_PERSON  = 2

# تنظیمات نمودار
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"
CMAP_FACE     = "gray"
STYLE         = "seaborn-v0_8-whitegrid"