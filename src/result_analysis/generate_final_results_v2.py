"""
Final Results Generation Script (V2)

Author : Gargi Kedia
Version : V2
"""

import pandas as pd
import numpy as np

from config import *

from plot_utils_v2 import (
    plot_horizontal_bar,
    plot_scatter,
    plot_grouped_metrics
)
# ==========================================================
# DISPLAY NAME MAPPING
# ==========================================================

DISPLAY_NAMES = {

    "efficientnet":
        "EfficientNet-B0",

    "mobilenet":
        "MobileNetV3",

    "vit":
        "Vision Transformer",

    "tabular_mlp":
        "Tabular MLP",

    "tabular_transformer":
        "Tabular Transformer",

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


# ==========================================================
# CATEGORY
# ==========================================================

def get_category(model):

    if model.startswith("fusion"):
        return "Fusion"

    if model.startswith("tabular"):
        return "Tabular"

    return "Image"


# ==========================================================
# LOAD METRICS
# ==========================================================

def load_metrics():

    print("\nLoading evaluation metrics...")

    rows = []

    for folder in sorted(RESULTS_DIR.iterdir()):

        if not folder.is_dir():
            continue

        if folder.name in [
            "plots",
            "final_results"
        ]:
            continue

        metrics_file = folder / "metrics.csv"

        if not metrics_file.exists():
            continue

        metric = pd.read_csv(
            metrics_file
        ).iloc[0]

        row = metric.to_dict()

        row["Model"] = folder.name

        row["Display Name"] = DISPLAY_NAMES.get(
            folder.name,
            folder.name
        )

        row["Category"] = get_category(
            folder.name
        )

        rows.append(row)

        print(f"Loaded : {folder.name}")

    df = pd.DataFrame(rows)

    print(
        f"\nLoaded {len(df)} models."
    )

    return df
# ==========================================================
# LOAD RASPBERRY PI BENCHMARK
# ==========================================================

def load_raspberry_pi():

    if not RPI_BENCHMARK.exists():
        print("Raspberry Pi benchmark not found.")
        return pd.DataFrame()

    df = pd.read_csv(RPI_BENCHMARK)

    model_map = {
        "ViT_Transformer": "ViT + Transformer",
        "ViT_MLP": "ViT + MLP",
        "EfficientNet_Transformer": "EfficientNet + Transformer",
        "EfficientNet_MLP": "EfficientNet + MLP",
        "MobileNet_Transformer": "MobileNet + Transformer",
        "MobileNet_MLP": "MobileNet + MLP",
    }

    df["Display Name"] = df["Model"].map(model_map)

    df.drop(columns=["Model"], inplace=True)

    return df


# ==========================================================
# LOAD CORAL BENCHMARK
# ==========================================================

def load_coral():

    if not CORAL_BENCHMARK.exists():
        print("Coral benchmark not found.")
        return pd.DataFrame()

    df = pd.read_csv(CORAL_BENCHMARK)

    model_map = {
        "ViT_Transformer": "ViT + Transformer",
        "ViT_MLP": "ViT + MLP",
        "EfficientNet_Transformer": "EfficientNet + Transformer",
        "EfficientNet_MLP": "EfficientNet + MLP",
        "MobileNet_Transformer": "MobileNet + Transformer",
        "MobileNet_MLP": "MobileNet + MLP",
    }

    df["Display Name"] = df["Model"].map(model_map)

    df.drop(columns=["Model"], inplace=True)

    return df

# ==========================================================
# MERGE EVERYTHING
# ==========================================================

def merge_results(metrics_df):

    merged = metrics_df.copy()

    rpi = load_raspberry_pi()

    if not rpi.empty:
        merged = merged.merge(
            rpi,
            on="Display Name",
            how="left"
        )

    coral = load_coral()

    if not coral.empty:
        merged = merged.merge(
            coral,
            on="Display Name",
            how="left",
            suffixes=("", "_Coral")
        )

    return merged
# ==========================================================
# SAVE COMPARISON TABLE
# ==========================================================

def save_comparison_table(df):

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    csv_file = OUTPUT_DIR / "comparison.csv"

    xlsx_file = OUTPUT_DIR / "comparison.xlsx"

    df.to_csv(

        csv_file,

        index=False

    )

    df.to_excel(

        xlsx_file,

        index=False
    )

    print("\nComparison table saved.")
# ==========================================================
# GENERATE EVALUATION METRIC GRAPHS
# ==========================================================

def generate_metric_graphs(df):

    print("\nGenerating evaluation graphs...")

    plot_horizontal_bar(
        df,
        metric="Accuracy",
        title="Model Accuracy Comparison",
        filename="accuracy_comparison",
        percentage=True
    )

    plot_horizontal_bar(
        df,
        metric="Precision (Weighted)",
        title="Weighted Precision Comparison",
        filename="precision_weighted",
        percentage=True
    )

    plot_horizontal_bar(
        df,
        metric="Recall (Weighted)",
        title="Weighted Recall Comparison",
        filename="recall_weighted",
        percentage=True
    )

    plot_horizontal_bar(
        df,
        metric="F1 (Weighted)",
        title="Weighted F1-score Comparison",
        filename="f1_weighted",
        percentage=True
    )

    plot_grouped_metrics(
        df,
        metrics=[
            "Accuracy",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1 (Weighted)"
        ],
        title="Overall Performance Comparison",
        filename="overall_metrics"
    )

    plot_horizontal_bar(
    df,
    metric="Accuracy",
    title="Model Accuracy Comparison",
    filename="accuracy_ranking",
    percentage=True
)


# ==========================================================
# RASPBERRY PI BENCHMARKS
# ==========================================================

def generate_rpi_graphs(df):

    print("Generating Raspberry Pi graphs...")

    rpi = df[
        df["Latency_ms"].notna()
    ].copy()

    if len(rpi) == 0:
        return

    plot_horizontal_bar(
        rpi,
        metric="Latency_ms",
        title="Latency Comparison (Raspberry Pi 5)",
        filename="rpi_latency",
        unit="ms"
    )

    plot_horizontal_bar(
        rpi,
        metric="FPS",
        title="FPS Comparison (Raspberry Pi 5)",
        filename="rpi_fps",
        unit="FPS"
    )

    plot_horizontal_bar(
        rpi,
        metric="RAM_MB",
        title="RAM Usage (Raspberry Pi 5)",
        filename="rpi_ram",
        unit="MB"
    )

    plot_horizontal_bar(
        rpi,
        metric="Estimated_Energy_J",
        title="Estimated Energy (Raspberry Pi 5)",
        filename="rpi_energy",
        unit="J"
    )
    plot_horizontal_bar(
    rpi,
    metric="Size_MB",
    title="Model Size (Raspberry Pi)",
    filename="rpi_model_size",
    percentage=False,
    unit="MB"
)

    plot_horizontal_bar(
    rpi,
    metric="Load_Time_s",
    title="Load Time (Raspberry Pi)",
    filename="rpi_load_time",
    percentage=False,
    unit="s"
)

    plot_horizontal_bar(
    rpi,
    metric="Runtime_s",
    title="Runtime (Raspberry Pi)",
    filename="rpi_runtime",
    percentage=False,
    unit="s"
)
print("RPi graph function finished.")

# ==========================================================
# CORAL BENCHMARKS
# ==========================================================

def generate_coral_graphs(df):

    print("Generating Coral graphs...")

    coral = df[
        df["Latency_ms_Coral"].notna()
    ].copy()

    if len(coral) == 0:
        return

    plot_horizontal_bar(
        coral,
        metric="Latency_ms_Coral",
        title="Latency Comparison (Coral TPU)",
        filename="coral_latency",
        unit="ms"
    )

    plot_horizontal_bar(
        coral,
        metric="FPS_Coral",
        title="FPS Comparison (Coral TPU)",
        filename="coral_fps",
        unit="FPS"
    )

    plot_horizontal_bar(
        coral,
        metric="RAM_MB_Coral",
        title="RAM Usage (Coral TPU)",
        filename="coral_ram",
        unit="MB"
    )

    plot_horizontal_bar(
        coral,
        metric="Estimated_Energy_J_Coral",
        title="Estimated Energy (Coral TPU)",
        filename="coral_energy",
        unit="J"
    )
    plot_horizontal_bar(
    coral,
    metric="Size_MB_Coral",
    title="Model Size (Coral TPU)",
    filename="coral_model_size",
    percentage=False,
    unit="MB"
)

    plot_horizontal_bar(
    coral,
    metric="Load_Time_s_Coral",
    title="Load Time (Coral TPU)",
    filename="coral_load_time",
    percentage=False,
    unit="s"
)

    plot_horizontal_bar(
    coral,
    metric="Runtime_s_Coral",
    title="Runtime (Coral TPU)",
    filename="coral_runtime",
    percentage=False,
    unit="s"
)
print("Coral graph function finished.")
# ==========================================================
# TRADEOFF PLOTS
# ==========================================================

def generate_tradeoff_graphs(df):

    print("\nGenerating trade-off graphs...")

    fusion = df[
        df["Category"] == "Fusion"
    ].copy()

    # Raspberry Pi

    plot_scatter(
        fusion,
        x="Latency_ms",
        y="Accuracy",
        title="Accuracy vs Latency (Raspberry Pi 5)",
        filename="tradeoff/accuracy_vs_latency_rpi",
        percentage_y=True
    )

    plot_scatter(
        fusion,
        x="FPS",
        y="Accuracy",
        title="Accuracy vs FPS (Raspberry Pi 5)",
        filename="tradeoff/accuracy_vs_fps_rpi",
        percentage_y=True
    )

    plot_scatter(
        fusion,
        x="Estimated_Energy_J",
        y="Accuracy",
        title="Accuracy vs Energy (Raspberry Pi 5)",
        filename="tradeoff/accuracy_vs_energy_rpi",
        percentage_y=True
    )

    plot_scatter(
        fusion,
        x="Size_MB",
        y="Accuracy",
        title="Accuracy vs Model Size",
        filename="tradeoff/accuracy_vs_model_size",
        percentage_y=True
    )

    # Coral

    plot_scatter(
        fusion,
        x="Latency_ms_Coral",
        y="Accuracy",
        title="Accuracy vs Latency (Coral TPU)",
        filename="tradeoff/accuracy_vs_latency_coral",
        percentage_y=True
    )

    plot_scatter(
        fusion,
        x="FPS_Coral",
        y="Accuracy",
        title="Accuracy vs FPS (Coral TPU)",
        filename="tradeoff/accuracy_vs_fps_coral",
        percentage_y=True
    )

    plot_scatter(
        fusion,
        x="Estimated_Energy_J_Coral",
        y="Accuracy",
        title="Accuracy vs Energy (Coral TPU)",
        filename="tradeoff/accuracy_vs_energy_coral",
        percentage_y=True
    )


# ==========================================================
# RANKING
# ==========================================================

def generate_ranking(df):

    ranking = df.copy()

    ranking["Overall Score"] = (

        0.40 * ranking["Accuracy"]

        + 0.20 * ranking["F1 (Weighted)"]

        + 0.20 * ranking["Precision (Weighted)"]

        + 0.20 * ranking["Recall (Weighted)"]

    )

    ranking = ranking.sort_values(

        "Overall Score",

        ascending=False

    )

    ranking["Rank"] = range(

        1,

        len(ranking)+1

    )

    ranking.to_csv(

        OUTPUT_DIR / "overall_ranking.csv",

        index=False

    )

    return ranking


# ==========================================================
# SUMMARY
# ==========================================================

def generate_summary(df):

    best = df.sort_values(

        "Overall Score",

        ascending=False

    ).iloc[0]

    with open(

        OUTPUT_DIR / "summary.txt",

        "w"

    ) as f:

        f.write(

            "FINAL MODEL SUMMARY\n"

        )

        f.write(

            "="*60 + "\n\n"

        )

        f.write(

            f"Best Overall Model : {best['Display Name']}\n"

        )

        f.write(

            f"Accuracy : {best['Accuracy']*100:.2f}%\n"

        )

        f.write(

            f"Weighted F1 : {best['F1 (Weighted)']*100:.2f}%\n"

        )

        f.write(

            f"Weighted Precision : {best['Precision (Weighted)']*100:.2f}%\n"

        )

        f.write(

            f"Weighted Recall : {best['Recall (Weighted)']*100:.2f}%\n"

        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    metrics = load_metrics()

    merged = merge_results(metrics)

    save_comparison_table(

        merged

    )

    generate_metric_graphs(

        merged

    )

    generate_rpi_graphs(

        merged

    )

    generate_coral_graphs(

        merged

    )

    generate_tradeoff_graphs(

        merged

    )

    ranking = generate_ranking(

        merged

    )

    generate_summary(

        ranking

    )

    print("\nDone.")


if __name__ == "__main__":

    main()