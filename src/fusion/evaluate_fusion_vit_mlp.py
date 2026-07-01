import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate_metrics import save_metrics

from src.fusion.fusion_dataset import FusionDataset
from src.fusion.fusion_model_vit_mlp import FusionViTMLP


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)


# =====================================================
# SCALER
# =====================================================

scaler = joblib.load(
    "models/fusion_scaler.pkl"
)


# =====================================================
# TEST TRANSFORM
# =====================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# TEST DATASET
# =====================================================

test_dataset = FusionDataset(
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images",
    scaler=scaler,
    transform=test_transform
)

print(f"Test Dataset Loaded: {len(test_dataset)} samples")


# =====================================================
# DATALOADER
# =====================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


# =====================================================
# MODEL
# =====================================================

model = FusionViTMLP()

model.load_state_dict(
    torch.load(
        "models/fusion_vit_mlp_40epoch.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Fusion ViT + MLP Model Loaded")


# =====================================================
# LOSS
# =====================================================

criterion = nn.CrossEntropyLoss()


# =====================================================
# EVALUATION
# =====================================================

correct = 0
total = 0
total_loss = 0.0

num_classes = 6

class_correct = [0] * num_classes
class_total = [0] * num_classes

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, tabular, labels in test_loader:

        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.long().to(device)

        outputs = model(images, tabular)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):

            class_correct[label] += (pred == label).item()
            class_total[label] += 1


# =====================================================
# RESULTS
# =====================================================

avg_loss = total_loss / len(test_loader)
accuracy = 100 * correct / total

print("\n" + "=" * 50)
print("Fusion ViT + MLP Evaluation")
print("=" * 50)

print(f"Test Loss     : {avg_loss:.4f}")
print(f"Test Accuracy : {accuracy:.2f}%")

print("=" * 50)

print("\nPer-Class Accuracy\n")

for i in range(num_classes):

    if class_total[i] > 0:

        class_acc = (
            100 * class_correct[i]
            / class_total[i]
        )

        bar = (
            "█" * int(class_acc // 5)
            + "░" * (20 - int(class_acc // 5))
        )

        print(
            f"Class {i}: "
            f"{bar} "
            f"{class_acc:.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )

    else:

        print(f"Class {i}: No Samples")


# =====================================================
# SAVE METRICS
# =====================================================

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
    model_name="fusion_vit_mlp"
)

print("\nAll evaluation files saved successfully.")