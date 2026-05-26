import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from image_dataset import ImageDataset
from vit_model import ViTClassifier


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# DATASET
dataset = ImageDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images"
)

print("Dataset Loaded")


# DATALOADER
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("DataLoader Ready")


# MODEL
model = ViTClassifier(num_classes=6)

model = model.to(device)

print("Model Loaded")


# LOSS FUNCTION
criterion = nn.CrossEntropyLoss()


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Training Starting...")


# TRAIN LOOP
epochs = 10

for epoch in range(epochs):

    total_loss = 0

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.long().to(device)

        # FORWARD
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)

             # BACKWARD
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

print("Training Complete")

torch.save(
    model.state_dict(),
    "models/vit_image_model_15epoch.pth"
)

print("Model Saved")