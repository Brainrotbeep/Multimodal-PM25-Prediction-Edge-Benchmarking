import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.fusion.fusion_dataset import FusionDataset
from src.fusion.fusion_model_efficientnet_featuregate import FusionEfficientNetFeatureGate

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# SCALER — train wala load karo
scaler = joblib.load("models/fusion_scaler.pkl")

# TEST TRANSFORM — sirf resize + normalize, koi augmentation nahi
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# TEST DATASET
test_dataset = FusionDataset(
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images",
    scaler=scaler,              # train ka scaler pass karo
    transform=test_transform    # koi augmentation nahi
)

print(f"Test Dataset Loaded: {len(test_dataset)} samples")

# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# MODEL
model = FusionEfficientNetFeatureGate()

model.load_state_dict(
    torch.load(
        "models/fusion_efficientnet_featuregate_20epoch.pth",
        map_location=device,
        weights_only=True       # warning nahi ayegi
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
class_total   = [0] * num_classes

criterion = nn.CrossEntropyLoss()
total_loss = 0.0

with torch.no_grad():

    for images, tabular, labels in test_loader:

        images  = images.to(device)
        tabular = tabular.to(device)
        labels  = labels.long().to(device)

        outputs = model(images, tabular)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            class_correct[label] += (pred == label).item()
            class_total[label]   += 1

# RESULTS
avg_loss = total_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"\n{'='*40}")
print(f"Test Loss:     {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"{'='*40}")

print("\nPer-Class Accuracy:")
for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        bar = "█" * int(class_acc // 5) + "░" * (20 - int(class_acc // 5))
        print(f"  Class {i}: {bar} {class_acc:.2f}%  ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"  Class {i}: no samples")