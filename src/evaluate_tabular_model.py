import torch

from torch.utils.data import DataLoader

from tabular_dataset import TabularDataset
from tabular_model import TabularTransformer


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# TEST DATASET
test_dataset = TabularDataset(
    csv_file="data/clean_test_metadata.csv"
)

print("Test Dataset Loaded")


# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# MODEL
model = TabularTransformer()

model.load_state_dict(
    torch.load(
        "models/tabular_transformer_35epoch.pth"
    )
)

model = model.to(device)

model.eval()

print("Model Loaded")


# EVALUATION
correct = 0
total = 0


with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        labels = labels.to(device)

        outputs = model(features)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")