import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tabular_dataset import TabularDataset
from tabular_model import TabularTransformer


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# DATASET
dataset = TabularDataset(
    csv_file="data/clean_metadata.csv"
)

print("Dataset Loaded")


# DATALOADER
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("DataLoader Ready")


# MODEL
model = TabularTransformer()

model = model.to(device)

print("Model Loaded")


# LOSS
criterion = nn.CrossEntropyLoss()


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training Starting...")


# TRAIN LOOP
epochs = 35

for epoch in range(epochs):

    total_loss = 0

    correct = 0
    total = 0

    for features, labels in loader:

        features = features.to(device)

        labels = labels.to(device)

        # FORWARD
        outputs = model(features)

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
    "models/tabular_transformer_35epoch.pth"
)

print("Model Saved")