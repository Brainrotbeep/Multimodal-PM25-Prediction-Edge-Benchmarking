import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tabular.tabular_dataset import TabularDataset
from src.tabular.tabular_mlp_model import TabularMLP


# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using Device:", device)

# SCALER
scaler = joblib.load(
    "models/tabular_scaler.pkl"
)

# TEST DATASET
test_dataset = TabularDataset(
    csv_file="data/clean_test_metadata.csv",
    scaler=scaler
)

print(
    f"Test Dataset Loaded: "
    f"{len(test_dataset)} samples"
)

# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# MODEL
model = TabularMLP()

model.load_state_dict(
    torch.load(
        "models/tabular_mlp_40epoch.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model Loaded")

# EVALUATION
correct = 0
total = 0

num_classes = 6

class_correct = [0] * num_classes
class_total = [0] * num_classes

criterion = nn.CrossEntropyLoss()
total_loss = 0.0

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.long().to(device)

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        for label, pred in zip(
            labels,
            predicted
        ):

            class_correct[label] += (
                pred == label
            ).item()

            class_total[label] += 1

# RESULTS
avg_loss = (
    total_loss / len(test_loader)
)

accuracy = (
    100 * correct / total
)

print(f"\n{'='*40}")
print(f"Test Loss:     {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"{'='*40}")

print("\nPer-Class Accuracy:")

for i in range(num_classes):

    if class_total[i] > 0:

        class_acc = (
            100 * class_correct[i]
            / class_total[i]
        )

        bar = (
            "█" * int(class_acc // 5)
            +
            "░" * (
                20 - int(class_acc // 5)
            )
        )

        print(
            f"  Class {i}: "
            f"{bar} "
            f"{class_acc:.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )

    else:

        print(
            f"  Class {i}: no samples"
        )