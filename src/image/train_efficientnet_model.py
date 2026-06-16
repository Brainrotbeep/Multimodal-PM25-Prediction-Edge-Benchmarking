import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.image.image_dataset import ImageDataset
from src.image.efficientnet_model import EfficientNetClassifier


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
train_dataset = ImageDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images",
    transform=train_transform
)

print(f"Train Dataset Loaded: {len(train_dataset)} samples")


# DATALOADER
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


# MODEL
model = EfficientNetClassifier(
    num_classes=6
).to(device)

model.load_state_dict(
    torch.load(
        "models/efficientnet_b0_20epoch.pth",
        map_location=device,
        weights_only=True
    )
)

print("Checkpoint Loaded")


# LOSS
class_counts = [
    1780,
    1780,
    1780,
    1780,
    822,
    356
]

class_weights = torch.tensor(
    [1.0 / c for c in class_counts],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# OPTIMIZER
optimizer = torch.optim.AdamW([
    {
        "params": [
            p for n, p in model.backbone.named_parameters()
            if "blocks" in n and p.requires_grad
        ],
        "lr": 5e-6
    },
    {
        "params": model.backbone.classifier.parameters(),
        "lr": 5e-5
    }
],
weight_decay=1e-4)


# RESUME SETTINGS
start_epoch = 20
epochs = 40


# SCHEDULER
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[5e-6, 5e-5],
    steps_per_epoch=len(train_loader),
    epochs=epochs - start_epoch,
    pct_start=0.1
)

print("Resume Training Starting...")


# TRAIN LOOP
for epoch in range(start_epoch, epochs):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    avg_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {train_acc:.2f}%"
    )

    # SAVE EVERY 5 EPOCHS
    if (epoch + 1) % 5 == 0:

        torch.save(
            model.state_dict(),
            f"models/efficientnet_b0_epoch_{epoch+1}.pth"
        )

        print(
            f"Checkpoint Saved: Epoch {epoch+1}"
        )


print("\nTraining Complete")


torch.save(
    model.state_dict(),
    "models/efficientnet_b0_40epoch_resume.pth"
)

print("Final Model Saved")