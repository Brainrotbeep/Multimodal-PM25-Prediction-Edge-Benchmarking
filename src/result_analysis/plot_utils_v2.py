"""
Publication-quality plotting utilities.

Author : Gargi Kedia
Version : V2
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

from config import *

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"

# ==========================================================
# CONSTANTS
# ==========================================================

CATEGORY_ORDER = [
    "Image",
    "Tabular",
    "Fusion"
]

CATEGORY_COLORS = {
    "Image": "#4C72B0",
    "Tabular": "#55A868",
    "Fusion": "#C44E52"
}

# ==========================================================
# OUTPUT
# ==========================================================

def save_figure(fig, relative_path):

    output_path = OUTPUT_DIR / relative_path
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        str(output_path) + ".png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

# ==========================================================
# COLORS
# ==========================================================

def get_colors(df):

    return [
        CATEGORY_COLORS.get(
            c,
            "#808080"
        )
        for c in df["Category"]
    ]

# ==========================================================
# LEGEND
# ==========================================================

def add_category_legend(ax):

    handles = []

    for cat in CATEGORY_ORDER:

        handles.append(

            Line2D(

                [0],

                [0],

                marker="s",

                color="w",

                label=cat,

                markerfacecolor=CATEGORY_COLORS[cat],

                markersize=12

            )

        )

    ax.legend(

        handles=handles,

        fontsize=11,

        frameon=False,

        loc="upper left"

    )

# ==========================================================
# FORMATTERS
# ==========================================================

def format_number(value, unit=None):

    if value is None:
        return ""

    if unit == "%":
        return f"{value:.2f}%"

    if unit == "ms":
        return f"{value:.1f} ms"

    if unit == "MB":
        return f"{value:.2f} MB"

    if unit == "FPS":
        return f"{value:.2f}"

    if unit == "J":
        return f"{value:,.1f}"

    if unit == "s":
        return f"{value:.2f} s"

    return f"{value:.2f}"

# ==========================================================
# AXIS LABELS
# ==========================================================

AXIS_LABELS = {

    "Accuracy":
        "Accuracy (%)",

    "Precision (Weighted)":
        "Weighted Precision (%)",

    "Precision (Macro)":
        "Macro Precision (%)",

    "Recall (Weighted)":
        "Weighted Recall (%)",

    "Recall (Macro)":
        "Macro Recall (%)",

    "F1 (Weighted)":
        "Weighted F1 (%)",

    "F1 (Macro)":
        "Macro F1 (%)",

    "Latency_ms":
        "Latency (ms)",

    "Latency_ms_Coral":
        "Latency (ms)",

    "FPS":
        "FPS",

    "FPS_Coral":
        "FPS",

    "RAM_MB":
        "RAM Usage (MB)",

    "RAM_MB_Coral":
        "RAM Usage (MB)",

    "Estimated_Energy_J":
        "Estimated Energy (J)",

    "Estimated_Energy_J_Coral":
        "Estimated Energy (J)",

    "Runtime_s":
        "Runtime (s)",

    "Runtime_s_Coral":
        "Runtime (s)",

    "Load_Time_s":
        "Load Time (s)",

    "Load_Time_s_Coral":
        "Load Time (s)",

    "Size_MB":
        "Model Size (MB)",

    "Size_MB_Coral":
        "Model Size (MB)"
}

# ==========================================================
# METRIC TYPE
# ==========================================================

PERCENTAGE_COLUMNS = {

    "Accuracy",

    "Precision (Weighted)",

    "Precision (Macro)",

    "Recall (Weighted)",

    "Recall (Macro)",

    "F1 (Weighted)",

    "F1 (Macro)"

}
# ==========================================================
# PUBLICATION QUALITY HORIZONTAL BAR PLOT
# ==========================================================

def plot_horizontal_bar(
    df,
    metric,
    title,
    filename,
    xlabel=None,
    percentage=False,
    unit=None
):
    """
    Generic horizontal bar chart.

    Parameters
    ----------
    percentage : bool
        True for evaluation metrics (Accuracy, Precision, Recall, F1)

    unit :
        None
        "ms"
        "MB"
        "FPS"
        "J"
        "s"
    """

    plot_df = df.copy()

    # ------------------------------------------
    # Remove NaN values
    # ------------------------------------------

    plot_df = plot_df[
        plot_df[metric].notna()
    ]

    if len(plot_df) == 0:
        return

    # ------------------------------------------
    # Convert to percentage
    # ------------------------------------------

    if percentage:

        plot_df[metric] = (
            plot_df[metric] * 100
        )

    # ------------------------------------------
    # Sort
    # ------------------------------------------

    plot_df = plot_df.sort_values(
        metric,
        ascending=True
    )

    colors = get_colors(plot_df)

    fig, ax = plt.subplots(
        figsize=(14, 7.5)
    )

    bars = ax.barh(

        plot_df["Display Name"],

        plot_df[metric],

        color=colors,

        edgecolor="black",

        linewidth=0.5,

        height=0.65

    )

    # ------------------------------------------
    # Axis Labels
    # ------------------------------------------

    ax.set_title(

        title,

        fontsize=18,

        fontweight="bold",

        pad=15

    )

    if xlabel is None:

        xlabel = AXIS_LABELS.get(
            metric,
            metric
        )

    ax.set_xlabel(

        xlabel,

        fontsize=13

    )

    ax.tick_params(

        axis="y",

        labelsize=11

    )

    ax.tick_params(

        axis="x",

        labelsize=11

    )

    # ------------------------------------------
    # Grid
    # ------------------------------------------

    ax.grid(

        axis="x",

        linestyle="--",

        alpha=0.35

    )

    ax.set_axisbelow(True)

    # ------------------------------------------
    # Smart X limits
    # ------------------------------------------

    max_value = plot_df[metric].max()

    margin = max_value * 0.20

    ax.set_xlim(
    0,
    max_value + margin
)

    # ------------------------------------------
    # Percentage axis
    # ------------------------------------------

    if percentage:

        ax.xaxis.set_major_formatter(

            mtick.PercentFormatter()

        )

    # ------------------------------------------
    # Value Labels
    # ------------------------------------------

    offset = max_value * 0.015

    for bar, value in zip(

        bars,

        plot_df[metric]

    ):

        if percentage:

            label = f"{value:.2f}%"

        else:

            label = format_number(

                value,

                unit

            )

        ax.text(

            value + offset,

            bar.get_y()
            + bar.get_height()/2,

            label,

            va="center",

            fontsize=10,

            fontweight="bold"

        )

    # ------------------------------------------
    # Legend
    # ------------------------------------------

    add_category_legend(ax)

    # ------------------------------------------
    # Remove top/right borders
    # ------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_figure(

        fig,

        filename

    )
# ==========================================================
# PUBLICATION QUALITY SCATTER PLOT
# ==========================================================

def plot_scatter(
    df,
    x,
    y,
    title,
    filename,
    x_label=None,
    y_label=None,
    percentage_y=False,
    percentage_x=False
):
    """
    Publication-quality scatter plot.

    Parameters
    ----------
    percentage_y : bool
        Convert Y values (0-1) into percentage

    percentage_x : bool
        Convert X values (0-1) into percentage
    """

    plot_df = df.copy()

    plot_df = plot_df[
        plot_df[x].notna() &
        plot_df[y].notna()
    ]

    if len(plot_df) == 0:
        return

    # ------------------------------------------
    # Convert percentages
    # ------------------------------------------

    if percentage_x:
        plot_df[x] *= 100

    if percentage_y:
        plot_df[y] *= 100

    colors = get_colors(plot_df)

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.scatter(
        plot_df[x],
        plot_df[y],
        s=130,
        c=colors,
        edgecolors="black",
        linewidths=0.7,
        alpha=0.9,
        zorder=3
    )

    # ------------------------------------------
    # Dynamic annotation offsets
    # ------------------------------------------

    offsets = [

    (12,12),

    (-12,12),

    (12,-12),

    (-12,-12),

    (18,0),

    (-18,0),

    (0,18),

    (0,-18),

    (22,10),

    (-22,10),

    (22,-10),

    (-22,-10)

]

    for i, (_, row) in enumerate(plot_df.iterrows()):

        dx, dy = offsets[i % len(offsets)]

        ax.annotate(
            row["Display Name"],
            (row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="gray",
                alpha=0.85
            ),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.5,
                color="gray"
            )
        )

    # ------------------------------------------
    # Labels
    # ------------------------------------------

    if x_label is None:
        x_label = AXIS_LABELS.get(x, x)

    if y_label is None:
        y_label = AXIS_LABELS.get(y, y)

    ax.set_xlabel(
        x_label,
        fontsize=13
    )

    ax.set_ylabel(
        y_label,
        fontsize=13
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    # ------------------------------------------
    # Percentage Axis
    # ------------------------------------------

    if percentage_x:
        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter()
        )

    if percentage_y:
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter()
        )

    # ------------------------------------------
    # Grid
    # ------------------------------------------

    ax.grid(
        linestyle="--",
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # ------------------------------------------
    # Smart margins
    # ------------------------------------------

    x_min = plot_df[x].min()
    x_max = plot_df[x].max()

    y_min = plot_df[y].min()
    y_max = plot_df[y].max()

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    ax.set_xlim(
        x_min - x_pad,
        x_max + x_pad
    )

    ax.set_ylim(
        y_min - y_pad,
        y_max + y_pad
    )

    # ------------------------------------------
    # Clean borders
    # ------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------
    # Category legend
    # ------------------------------------------

    add_category_legend(ax)

    plt.tight_layout()

    save_figure(
        fig,
        filename
    )
# ==========================================================
# PUBLICATION QUALITY GROUPED METRIC COMPARISON
# ==========================================================

def plot_grouped_metrics(
    df,
    metrics,
    title,
    filename
):
    """
    Publication-quality grouped metric comparison.

    All metrics are converted to percentage automatically.
    """

    plot_df = df.copy()

    # ------------------------------------------
    # Sort by Accuracy
    # ------------------------------------------

    plot_df = plot_df.sort_values(
        "Accuracy",
        ascending=False
    )

    # ------------------------------------------
    # Convert to %
    # ------------------------------------------

    for metric in metrics:
        plot_df[metric] *= 100

    fig, ax = plt.subplots(
        figsize=(18, 8)
    )

    x = list(range(len(plot_df)))

    width = 0.11

    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#CCB974",
        "#64B5CD",
        "#8C8C8C"
    ]

    # ------------------------------------------
    # Bars
    # ------------------------------------------

    for i, metric in enumerate(metrics):

        positions = [
            k + i * width
            for k in x
        ]

        ax.bar(
            positions,
            plot_df[metric],
            width=width,
            label=metric,
            color=colors[i],
            edgecolor="black",
            linewidth=0.3
        )

    # ------------------------------------------
    # X ticks
    # ------------------------------------------

    center = [
        k + width * (len(metrics)-1)/2
        for k in x
    ]

    ax.set_xticks(center)

    ax.set_xticklabels(
        plot_df["Display Name"],
        rotation=30,
        ha="right",
        fontsize=10
    )

    # ------------------------------------------
    # Labels
    # ------------------------------------------

    ax.set_ylabel(
        "Performance (%)",
        fontsize=13
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter()
    )

    # ------------------------------------------
    # Grid
    # ------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # ------------------------------------------
    # Legend
    # ------------------------------------------

    ax.legend(
        fontsize=10,
        frameon=False,
        ncol=2,
        loc="upper right"
    )

    # ------------------------------------------
    # Remove borders
    # ------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_figure(
        fig,
        filename
    )


# ==========================================================
# END OF FILE
# ==========================================================