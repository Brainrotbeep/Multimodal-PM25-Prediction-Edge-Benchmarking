import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def save_metrics(
    true_labels,
    predictions,
    class_names,
    model_name,
    save_root="results"
):
    """
    Saves:
    - metrics.csv
    - classification_report.txt
    - confusion_matrix.png
    """

    save_dir = os.path.join(save_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # ==========================
    # Metrics
    # ==========================

    accuracy = accuracy_score(true_labels, predictions)

    precision_weighted = precision_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_weighted = recall_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall_macro = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = pd.DataFrame({

        "Accuracy":[accuracy],

        "Precision (Weighted)":[precision_weighted],

        "Precision (Macro)":[precision_macro],

        "Recall (Weighted)":[recall_weighted],

        "Recall (Macro)":[recall_macro],

        "F1 (Weighted)":[f1_weighted],

        "F1 (Macro)":[f1_macro]

    })

    metrics.to_csv(
        os.path.join(save_dir, "metrics.csv"),
        index=False
    )

    # ==========================
    # Classification Report
    # ==========================

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    with open(
        os.path.join(save_dir, "classification_report.txt"),
        "w"
    ) as f:

        f.write(report)

    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "confusion_matrix.png"),
        dpi=300
    )

    plt.close()

    # ==========================
    # Print Summary
    # ==========================

    print("\n==============================")
    print(model_name)
    print("==============================")

    print(metrics)

    print("\nResults Saved Successfully")

    print(f"Location : {save_dir}")