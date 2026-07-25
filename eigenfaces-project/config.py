# Central configuration file
DATA_PATH = 'data/'
OUTPUT_PATH = 'outputs/'

import os

#  path
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
REPORT_DIR  = os.path.join(ROOT_DIR, "report")

#  data
IMAGE_SIZE      = 64          # ابعاد تصویر: 64x64
N_FEATURES      = 64 * 64     # = 4096
N_SAMPLES       = 400         # تعداد کل تصاویر
N_PERSONS       = 40          # تعداد افراد
N_IMAGES_PERSON = 10          # تصویر به ازای هر نفر
RANDOM_STATE    = 42

#  PCA 
K_VALUES_RECONSTRUCTION = [1, 5, 10, 20, 30, 50, 100, 150, 200, 300, 400]
K_VALUES_RECOGNITION    = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
K_DEFAULT               = 50

#  آستانه‌ها 
EPSILON              = 1e-10   # آستانه عددی برای تشخیص صفر
VARIANCE_THRESHOLDS  = [0.80, 0.90, 0.95, 0.99]

#  تقسیم داده 
N_TRAIN_PER_PERSON = 8
N_TEST_PER_PERSON  = 2

#  نمودار 
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"
COLORMAP_FACE = "gray"
COLORMAP_DEV  = "RdGray"