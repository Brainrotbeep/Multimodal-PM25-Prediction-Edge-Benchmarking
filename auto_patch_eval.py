from pathlib import Path

ROOT = Path("src")

IMPORT_LINE = "from evaluate_metrics import save_metrics"

CLASS_NAMES_BLOCK = '''
# ===========================================
# SAVE METRICS + REPORT + CONFUSION MATRIX
# ===========================================

class_names = [
    "0-50",
    "51-100",
    "101-150",
    "151-200",
    "201-300",
    "301-600"
]

save_metrics(
    true_labels=all_labels,
    predictions=all_predictions,
    class_names=class_names,
    model_name="{model_name}"
)
'''

for file in ROOT.rglob("evaluate*.py"):

    text = file.read_text(encoding="utf-8")

    # --------------------------------------
    # Add Import
    # --------------------------------------

    if IMPORT_LINE not in text:

        text = IMPORT_LINE + "\n" + text

    # --------------------------------------
    # Remove sklearn confusion matrix import
    # --------------------------------------

    text = text.replace(
        "from sklearn.metrics import confusion_matrix\n",
        ""
    )

    # --------------------------------------
    # Skip already patched files
    # --------------------------------------

    if "save_metrics(" in text:

        print(f"Skipped : {file}")

        continue

    # --------------------------------------
    # Remove old confusion matrix section
    # --------------------------------------

    marker = "# CONFUSION MATRIX"

    if marker in text:

        text = text.split(marker)[0]

    # --------------------------------------
    # Folder name
    # --------------------------------------

    model_name = file.stem.replace(
        "evaluate_",
        ""
    )

    text += CLASS_NAMES_BLOCK.format(
        model_name=model_name
    )

    file.write_text(
        text,
        encoding="utf-8"
    )

    print(f"Updated : {file}")

print("\nDone!")