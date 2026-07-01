"""
Reusable plotting utilities for final benchmark analysis.

Author : Gargi Kedia
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import FuncFormatter

from config import *

plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["savefig.dpi"] = FIG_DPI
plt.rcParams["font.family"] = "DejaVu Sans"


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_figure(fig, filename):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if EXPORT_PNG:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

    if EXPORT_PDF:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.pdf",
            bbox_inches="tight"
        )

    if EXPORT_SVG:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.svg",
            bbox_inches="tight"
        )

    plt.close(fig)


# ==========================================================
# CATEGORY COLORS
# ==========================================================

def get_colors(df):

    colors = []

    for category in df["Category"]:

        colors.append(
            CATEGORY_COLORS.get(
                category,
                "#808080"
            )
        )

    return colors
def plot_horizontal_bar(
        df,
        metric,
        title,
        filename,
        xlabel=None,
        percentage=False,
        unit=""
):

    if df.empty or metric not in df.columns:
        return

    plot_df = df.sort_values(metric, ascending=True)

    colors = get_colors(plot_df)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    bars = ax.barh(
        plot_df["Display Name"],
        plot_df[metric],
        color=colors,
        height=BAR_HEIGHT
    )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    if xlabel is None:
        xlabel = AXIS_LABELS.get(metric, metric)

    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_SIZE
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.set_axisbelow(True)

    if percentage:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x*100:.0f}%")
        )

    max_value = plot_df[metric].max()
    offset = max_value * 0.02 if max_value > 0 else 0.02

    for bar, value in zip(bars, plot_df[metric]):

        if percentage:
            label = f"{value*100:.2f}%"
        else:
            label = f"{value:.2f}"
            if unit:
                label += f" {unit}"

        ax.text(
            value + offset,
            bar.get_y() + bar.get_height()/2,
            label,
            va="center",
            fontsize=ANNOTATION_SIZE
        )

    ax.set_xlim(0, max_value * 1.20)

    plt.tight_layout()

    save_figure(fig, filename)

def plot_scatter(
        df,
        x,
        y,
        title,
        filename,
        percentage_y=False
):

    if df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11,8)
    )

    colors = get_colors(df)

    ax.scatter(
        df[x],
        df[y],
        s=170,
        c=colors
    )

    offsets = [
        (12,12),
        (-12,12),
        (12,-12),
        (-12,-12),
        (18,0),
        (-18,0),
        (0,18),
        (0,-18)
    ]

    for i, (_, row) in enumerate(df.iterrows()):

        dx, dy = offsets[i % len(offsets)]

        ax.annotate(
            row["Display Name"],
            (row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9
        )

    ax.set_xlabel(
        AXIS_LABELS.get(x, x),
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        AXIS_LABELS.get(y, y),
        fontsize=LABEL_SIZE
    )

    if percentage_y:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v*100:.0f}%")
        )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.grid(
        linestyle="--",
        alpha=GRID_ALPHA
    )

    plt.tight_layout()

    save_figure(fig, filename)

def plot_grouped_metrics(
        df,
        metrics,
        title,
        filename
):

    if df.empty:
        return

    plot_df = df.sort_values(
        "Accuracy",
        ascending=False
    )

    fig, ax = plt.subplots(
        figsize=(18,8)
    )

    width = 0.18

    x = range(len(plot_df))

    for i, metric in enumerate(metrics):

        ax.bar(
            [k + i*width for k in x],
            plot_df[metric]*100,
            width,
            label=metric
        )

    ax.set_xticks(
        [k + width*1.5 for k in x]
    )

    ax.set_xticklabels(
        plot_df["Display Name"],
        rotation=30,
        ha="right"
    )

    ax.set_ylabel("Score (%)")

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.legend(fontsize=11)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    plt.tight_layout()

    save_figure(fig, filename)