import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.fusion.fusion_dataset import FusionDataset
from src.fusion.fusion_model_mobilenet_mlp import FusionMobileNetMLP


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DATASET
dataset = FusionDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images",
    scaler=None,
    transform=train_transform
)

joblib.dump(
    dataset.scaler,
    "models/fusion_scaler.pkl"
)

print("Scaler Saved")
print(f"Dataset Loaded: {len(dataset)} samples")

# DATALOADER
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# MODEL
model = FusionMobileNetMLP().to(device)
print("Model Loaded")

# LOSS
class_weights = torch.tensor(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        3.0
    ],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# OPTIMIZER
optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "mobilenet.blocks" in n and p.requires_grad
            ],
            "lr": 1e-5
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "mobilenet.blocks" not in n and p.requires_grad
            ],
            "lr": 3e-4
        }
    ],
    weight_decay=1e-4
)

# TRAINING SETTINGS
epochs = 40

# SCHEDULER
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 3e-4],
    steps_per_epoch=len(loader),
    epochs=epochs,
    pct_start=0.1
)

print("Training Starting...")

# TRAIN LOOP
for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, tabular, labels in loader:

        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images, tabular)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Loss: {avg_loss:.4f}  Acc: {accuracy:.2f}%"
    )
    if (epoch + 1) % 10 == 0:

        torch.save(
            model.state_dict(),
            f"models/fusion_mobilenet_mlp_epoch_{epoch+1}.pth"
        )

        print(
            f"Checkpoint Saved: Epoch {epoch+1}"
        )

print("Training Complete")

torch.save(
    model.state_dict(),
    "models/fusion_mobilenet_mlp_40epoch.pth"
)

print("Model Saved")