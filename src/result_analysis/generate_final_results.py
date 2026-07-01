"""
Final Benchmark Analysis

Reads:
    - All metrics.csv
    - Raspberry Pi benchmark
    - Coral benchmark

Generates:
    - comparison.csv
    - comparison.xlsx

Author : Gargi Kedia
"""

from pathlib import Path
import pandas as pd

from config import *

# ==========================================================
# BENCHMARK MODEL NAME MAPPING
# ==========================================================

BENCHMARK_MODEL_MAP = {

    "ViT_Transformer": "fusion_vit_transformer",
    "ViT_MLP": "fusion_vit_mlp",

    "EfficientNet_Transformer": "fusion_efficientnet_transformer",
    "EfficientNet_MLP": "fusion_efficientnet_mlp",

    "MobileNet_Transformer": "fusion_mobilenet_transformer",
    "MobileNet_MLP": "fusion_mobilenet_mlp"
}


# ==========================================================
# READ ALL METRICS
# ==========================================================

def load_metrics():

    rows = []

    print("=" * 60)
    print("Reading Metrics")
    print("=" * 60)

    for model_folder in sorted(RESULTS_DIR.iterdir()):

        if not model_folder.is_dir():
            continue

        if model_folder.name == "final_results":
            continue

        metrics_file = model_folder / "metrics.csv"

        if not metrics_file.exists():

            print(f"Skipping : {model_folder.name}")
            continue

        df = pd.read_csv(metrics_file)

        row = df.iloc[0].to_dict()

        row["Model"] = model_folder.name

        row["Display Name"] = MODEL_NAMES.get(
            model_folder.name,
            model_folder.name
        )

        row["Category"] = MODEL_CATEGORY.get(
            model_folder.name,
            "Unknown"
        )

        rows.append(row)

        print(f"Loaded : {model_folder.name}")

    metrics_df = pd.DataFrame(rows)

    print("\nModels Loaded :", len(metrics_df))

    return metrics_df


# ==========================================================
# READ BENCHMARK
# ==========================================================

def load_benchmark(file_path):

    if not file_path.exists():

        print(f"Benchmark file not found : {file_path}")
        return pd.DataFrame()

    benchmark = pd.read_csv(file_path)

    benchmark["Model"] = benchmark["Model"].replace(
        BENCHMARK_MODEL_MAP
    )

    return benchmark


# ==========================================================
# MERGE BENCHMARK
# ==========================================================

def merge_benchmark(metrics, benchmark, suffix):

    if benchmark.empty:
        return metrics

    merged = metrics.merge(

        benchmark,

        how="left",

        on="Model",

        suffixes=("", suffix)

    )

    return merged


# ==========================================================
# EXPORT TABLE
# ==========================================================

def export_table(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = OUTPUT_DIR / "comparison.csv"

    excel_path = OUTPUT_DIR / "comparison.xlsx"

    if EXPORT_CSV:

        df.to_csv(
            csv_path,
            index=False
        )

    if EXPORT_EXCEL:

        with pd.ExcelWriter(
                excel_path,
                engine="openpyxl"
        ) as writer:

            df.to_excel(

                writer,

                index=False,

                sheet_name="Comparison"

            )

    print("\nComparison Table Saved")

    print(csv_path)

    print(excel_path)
# ==========================================================
# IMPORT PLOT FUNCTIONS
# ==========================================================

from plot_utils import (
    plot_horizontal_bar,
    plot_scatter,
    plot_grouped_metrics
)


# ==========================================================
# PERFORMANCE GRAPHS
# ==========================================================

def generate_performance_graphs(df):

    print("\nGenerating Performance Graphs...")

    plot_horizontal_bar(
        df,
        metric="Accuracy",
        title="Model Accuracy Comparison",
        filename="accuracy_comparison",
        xlabel="Accuracy (%)"
    )

    plot_horizontal_bar(
        df,
        metric="Precision (Weighted)",
        title="Weighted Precision Comparison",
        filename="precision_weighted",
        xlabel="Weighted Precision (%)"
    )

    plot_horizontal_bar(
        df,
        metric="Precision (Macro)",
        title="Macro Precision Comparison",
        filename="precision_macro",
        xlabel="Macro Precision (%)"
    )

    plot_horizontal_bar(
        df,
        metric="Recall (Weighted)",
        title="Weighted Recall Comparison",
        filename="recall_weighted",
        xlabel="Weighted Recall (%)"
    )

    plot_horizontal_bar(
        df,
        metric="Recall (Macro)",
        title="Macro Recall Comparison",
        filename="recall_macro",
        xlabel="Macro Recall (%)"
    )

    plot_horizontal_bar(
        df,
        metric="F1 (Weighted)",
        title="Weighted F1 Comparison",
        filename="f1_weighted",
        xlabel="Weighted F1 (%)"
    )

    plot_horizontal_bar(
        df,
        metric="F1 (Macro)",
        title="Macro F1 Comparison",
        filename="f1_macro",
        xlabel="Macro F1 (%)"
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

    print("Performance graphs completed.")

# ==========================================================
# RASPBERRY PI BENCHMARK
# ==========================================================

def generate_rpi_graphs(df):

    rpi = df[df["Latency_ms"].notna()].copy()

    if len(rpi) == 0:

        print("No Raspberry Pi benchmark found.")

        return

    print("\nGenerating Raspberry Pi Graphs...")

    plot_horizontal_bar(

        rpi,

        metric="Latency_ms",

        title="Latency Comparison (Raspberry Pi)",

        filename="rpi_latency",

        xlabel="Latency (ms)",

        percentage=False

    )

    plot_horizontal_bar(

        rpi,

        metric="FPS",

        title="FPS Comparison (Raspberry Pi)",

        filename="rpi_fps",

        xlabel="Frames Per Second",

        percentage=False

    )

    plot_horizontal_bar(

        rpi,

        metric="RAM_MB",

        title="RAM Usage (Raspberry Pi)",

        filename="rpi_ram",

        xlabel="RAM (MB)",

        percentage=False

    )

    plot_horizontal_bar(

        rpi,

        metric="Size_MB",

        title="Model Size",

        filename="model_size",

        xlabel="Model Size (MB)",

        percentage=False

    )

    plot_horizontal_bar(

        rpi,

        metric="Load_Time_s",

        title="Load Time",

        filename="load_time",

        xlabel="Seconds",

        percentage=False

    )

    plot_horizontal_bar(

        rpi,

        metric="Estimated_Energy_J",

        title="Energy Consumption",

        filename="energy",

        xlabel="Estimated Energy (J)",

        percentage=False

    )

    print("Raspberry Pi graphs completed.")
# ==========================================================
# TRADEOFF PLOTS
# ==========================================================

def generate_tradeoff_graphs(df):

    rpi = df[df["Latency_ms"].notna()].copy()

    if len(rpi) == 0:

        return

    print("\nGenerating Trade-off Graphs...")

    plot_scatter(

        rpi,

        x="Latency_ms",

        y="Accuracy",

        title="Accuracy vs Latency",

        filename="accuracy_vs_latency"

    )

    plot_scatter(

        rpi,

        x="FPS",

        y="Accuracy",

        title="Accuracy vs FPS",

        filename="accuracy_vs_fps"

    )

    plot_scatter(

        rpi,

        x="Estimated_Energy_J",

        y="Accuracy",

        title="Accuracy vs Energy",

        filename="accuracy_vs_energy"

    )

    plot_scatter(

        rpi,

        x="Size_MB",

        y="Accuracy",

        title="Accuracy vs Model Size",

        filename="accuracy_vs_modelsize"

    )

    print("Trade-off graphs completed.")
# ==========================================================
# CORAL BENCHMARK
# ==========================================================

def generate_coral_graphs(df):

    coral = df[df["Latency_ms_Coral"].notna()].copy()

    if len(coral) == 0:

        print("No Coral Benchmark Found.")

        return

    print("\nGenerating Coral Graphs...")

    plot_horizontal_bar(
        coral,
        metric="Latency_ms_Coral",
        title="Latency Comparison (Coral TPU)",
        filename="coral_latency",
        xlabel="Latency (ms)",
        percentage=False
    )

    plot_horizontal_bar(
        coral,
        metric="FPS_Coral",
        title="FPS Comparison (Coral TPU)",
        filename="coral_fps",
        xlabel="Frames Per Second",
        percentage=False
    )

    plot_horizontal_bar(
        coral,
        metric="RAM_MB_Coral",
        title="RAM Usage (Coral TPU)",
        filename="coral_ram",
        xlabel="RAM (MB)",
        percentage=False
    )

    plot_horizontal_bar(
        coral,
        metric="Estimated_Energy_J_Coral",
        title="Estimated Energy (Coral TPU)",
        filename="coral_energy",
        xlabel="Energy (J)",
        percentage=False
    )

    print("Coral graphs completed.")
# ==========================================================
# OVERALL RANKING
# ==========================================================

def generate_ranking(df):

    ranking = df.copy()

    ranking["Overall Score"] = (

        ranking["Accuracy"] * 0.40

        +

        ranking["F1 (Weighted)"] * 0.30

        +

        ranking["F1 (Macro)"] * 0.20

        +

        ranking["Precision (Macro)"] * 0.10

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

        OUTPUT_DIR/"overall_ranking.csv",

        index=False

    )

    print("Overall Ranking Saved.")

    return ranking
# ==========================================================
# SUMMARY
# ==========================================================

def generate_summary(df):

    best_accuracy = df.loc[

        df["Accuracy"].idxmax()

    ]

    best_f1 = df.loc[

        df["F1 (Weighted)"].idxmax()

    ]

    summary = []

    summary.append("="*60)

    summary.append("FINAL MODEL SUMMARY")

    summary.append("="*60)

    summary.append("")

    summary.append(

        f"Highest Accuracy : "

        f"{best_accuracy['Display Name']} "

        f"({best_accuracy['Accuracy']:.2f}%)"

    )

    summary.append(

        f"Best Weighted F1 : "

        f"{best_f1['Display Name']} "

        f"({best_f1['F1 (Weighted)']:.2f})"

    )

    summary.append("")

    summary.append("Recommendation")

    summary.append("----------------------------")

    summary.append(

        "Highest Accuracy : EfficientNet + Transformer"

    )

    summary.append(

        "Best Edge Trade-off : MobileNet + Transformer"

    )

    summary.append(

        "Most Compact : MobileNet + MLP"

    )

    summary.append(

        "Highest Macro Performance : ViT + MLP"

    )

    with open(

        OUTPUT_DIR/"summary.txt",

        "w"

    ) as f:

        for line in summary:

            f.write(line+"\n")

    print("Summary Saved.")


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\nLoading Evaluation Metrics...\n")

    metrics = load_metrics()

    print(metrics[
        [
            "Display Name",
            "Accuracy"
        ]
    ])

    print("\nLoading Raspberry Pi Benchmark...")

    rpi = load_benchmark(
        RPI_BENCHMARK
    )

    print("Loading Coral Benchmark...")

    coral = load_benchmark(
        CORAL_BENCHMARK
    )

    merged = merge_benchmark(
        metrics,
        rpi,
        "_RPi"
    )

    merged = merge_benchmark(
        merged,
        coral,
        "_Coral"
    )

    export_table(
        merged
    )
    generate_performance_graphs(merged)

    generate_rpi_graphs(merged)

    generate_coral_graphs(merged)

    generate_tradeoff_graphs(merged)

    ranking = generate_ranking(merged)

    generate_summary(ranking)

    print("\nAnalysis Complete!")

    print("Results saved to:")

    print(OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":

    main()

