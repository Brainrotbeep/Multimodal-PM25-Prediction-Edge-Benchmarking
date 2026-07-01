"""
Configuration file for Final Benchmark Analysis

Author : Gargi Kedia
Project : Multimodal PM2.5 Prediction Edge Benchmarking
"""

from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS_DIR / "final_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# BENCHMARK FILES
# =====================================================

RPI_BENCHMARK = (
    PROJECT_ROOT
    / "raspberrypi5_benchmarks"
    / "benchmark_results_raspberrypi5_200x10.csv"
)

CORAL_BENCHMARK = (
    PROJECT_ROOT
    / "coral_benchmark"
    / "coral_benchmark_results.csv"
)

# =====================================================
# MODEL DISPLAY NAMES
# =====================================================

MODEL_NAMES = {

    # ---------------- IMAGE ----------------

    "efficientnet":
        "EfficientNet-B0",

    "mobilenet":
        "MobileNetV3",

    "vit":
        "Vision Transformer",

    # ---------------- TABULAR ----------------

    "tabular_mlp":
        "Tabular MLP",

    "tabular_transformer":
        "Tabular Transformer",

    # ---------------- FUSION ----------------

    "fusion_efficientnet_mlp":
        "EfficientNet + MLP",

    "fusion_efficientnet_transformer":
        "EfficientNet + Transformer",

    "fusion_mobilenet_mlp":
        "MobileNet + MLP",

    "fusion_mobilenet_transformer":
        "MobileNet + Transformer",

    "fusion_vit_mlp":
        "ViT + MLP",

    "fusion_vit_transformer":
        "ViT + Transformer"
}

# =====================================================
# MODEL CATEGORIES
# =====================================================

MODEL_CATEGORY = {

    "efficientnet": "Image",

    "mobilenet": "Image",

    "vit": "Image",

    "tabular_mlp": "Tabular",

    "tabular_transformer": "Tabular",

    "fusion_efficientnet_mlp": "Fusion",

    "fusion_efficientnet_transformer": "Fusion",

    "fusion_mobilenet_mlp": "Fusion",

    "fusion_mobilenet_transformer": "Fusion",

    "fusion_vit_mlp": "Fusion",

    "fusion_vit_transformer": "Fusion"
}
FUSION_MODELS = [

    "EfficientNet + Transformer",

    "EfficientNet + MLP",

    "ViT + Transformer",

    "ViT + MLP",

    "MobileNet + Transformer",

    "MobileNet + MLP"
]
# =====================================================
# COLORS
# =====================================================

CATEGORY_COLORS = {

    "Image": "#4C78A8",

    "Tabular": "#F58518",

    "Fusion": "#54A24B"
}

# =====================================================
# FIGURE SETTINGS
# =====================================================

FIG_DPI = 300

FIG_WIDTH = 14

FIG_HEIGHT = 8

TITLE_SIZE = 20

LABEL_SIZE = 16

TICK_SIZE = 13

LEGEND_SIZE = 13

ANNOTATION_SIZE = 11

GRID_ALPHA = 0.3

BAR_HEIGHT = 0.65

# =====================================================
# METRICS
# =====================================================

METRIC_COLUMNS = [

    "Accuracy",

    "Precision (Weighted)",

    "Precision (Macro)",

    "Recall (Weighted)",

    "Recall (Macro)",

    "F1 (Weighted)",

    "F1 (Macro)"
]

# =====================================================
# BENCHMARK COLUMNS
# =====================================================

BENCHMARK_COLUMNS = [

    "Size_MB",

    "Load_Time_s",

    "Latency_ms",

    "FPS",

    "RAM_MB",

    "Runtime_s",

    "Estimated_Energy_J"
]
# =====================================================
# AXIS LABELS
# =====================================================

AXIS_LABELS = {

    "Accuracy": "Accuracy (%)",

    "Precision (Weighted)": "Weighted Precision (%)",

    "Precision (Macro)": "Macro Precision (%)",

    "Recall (Weighted)": "Weighted Recall (%)",

    "Recall (Macro)": "Macro Recall (%)",

    "F1 (Weighted)": "Weighted F1-score (%)",

    "F1 (Macro)": "Macro F1-score (%)",

    "Latency_ms": "Latency (ms)",

    "Latency_ms_Coral": "Latency (ms)",

    "FPS": "Frames Per Second",

    "FPS_Coral": "Frames Per Second",

    "RAM_MB": "RAM Usage (MB)",

    "RAM_MB_Coral": "RAM Usage (MB)",

    "Runtime_s": "Runtime (s)",

    "Runtime_s_Coral": "Runtime (s)",

    "Estimated_Energy_J": "Energy (J)",

    "Estimated_Energy_J_Coral": "Energy (J)",

    "Size_MB": "Model Size (MB)",

    "Load_Time_s": "Load Time (s)"
}

# =====================================================
# EXPORT SETTINGS
# =====================================================

EXPORT_EXCEL = True

EXPORT_CSV = True

EXPORT_PDF = False

EXPORT_PNG = True

EXPORT_SVG = False