import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.image.image_dataset import ImageDataset
from src.image.vit_model import ViTClassifier

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# TEST TRANSFORM — sirf resize + normalize, koi augmentation nahi
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# TEST DATASET
test_dataset = ImageDataset(
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images",
    transform=test_transform
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
model = ViTClassifier(num_classes=6)

model.load_state_dict(
    torch.load(
        "models/vit_image_model_40epoch.pth",   # sahi file name
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
class_total   = [0] * num_classes

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.long().to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

        # per-class tracking
        for label, pred in zip(labels, predicted):
            class_correct[label] += (pred == label).item()
            class_total[label]   += 1

# RESULTS
accuracy = 100 * correct / total
print(f"\n{'='*40}")
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