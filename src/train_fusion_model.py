import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# DATASET
dataset = FusionDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images"
)

print("Dataset Loaded")


# DATALOADER
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("DataLoader Ready")


# MODEL
model = FusionModel()

model = model.to(device)

print("Model Loaded")


# LOSS
criterion = nn.CrossEntropyLoss()


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
    weight_decay=1e-4
)

print("Training Starting...")


# TRAIN LOOP
epochs = 15

for epoch in range(epochs):

    total_loss = 0

    correct = 0
    total = 0

    for images, tabular, labels in loader:

        images = images.to(device)

        tabular = tabular.to(device)

        labels = labels.to(device)

        # FORWARD
        outputs = model(
            images,
            tabular
        )

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        loss = criterion(
            outputs,
            labels
        )

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
    "models/fusion_model_10epoch.pth"
)

print("Model Saved")