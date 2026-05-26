import torch

from torch.utils.data import DataLoader

from image_dataset import ImageDataset
from vit_model import ViTClassifier


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# TEST DATASET
test_dataset = ImageDataset(
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images"
)

print("Test Dataset Loaded")


# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# MODEL
model = ViTClassifier(num_classes=6)

model.load_state_dict(
    torch.load("models/vit_image_model_15epoch.pth")
)

model = model.to(device)

model.eval()

print("Model Loaded")


# EVALUATION
correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.long().to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")