import torch

from torch.utils.data import DataLoader

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# TEST DATASET
test_dataset = FusionDataset(
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images"
)

print("Test Dataset Loaded")


# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# MODEL
model = FusionModel()

model.load_state_dict(
    torch.load(
        "models/fusion_model_10epoch.pth"
    )
)

model = model.to(device)

model.eval()

print("Model Loaded")


# EVALUATION
correct = 0
total = 0


with torch.no_grad():

    for images, tabular, labels in test_loader:

        images = images.to(device)

        tabular = tabular.to(device)

        labels = labels.to(device)

        outputs = model(
            images,
            tabular
        )

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")