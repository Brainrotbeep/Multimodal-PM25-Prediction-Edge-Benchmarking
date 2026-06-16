import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.tabular.tabular_dataset import TabularDataset
from src.tabular.tabular_model import TabularTransformer

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# DATASET
dataset = TabularDataset(
    csv_file="data/clean_metadata.csv",
    scaler=None             # train pe fit karega
)

# scaler save karo — eval mein use hoga
joblib.dump(dataset.scaler, "models/tabular_scaler.pkl")
print("Scaler Saved")

print(f"Dataset Loaded: {len(dataset)} samples")

# DATALOADER
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# MODEL
model = TabularTransformer().to(device)
print("Model Loaded")

# LOSS — weighted for class imbalance
class_counts = [1780, 1780, 1780, 1780, 822, 356]
class_weights = torch.tensor(
    [1.0 / c for c in class_counts],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

# OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# SCHEDULER
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(loader),
    epochs=40,
    pct_start=0.1
)

print("Training Starting...")

# TRAIN LOOP
epochs = 40

for epoch in range(epochs):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:

        features = features.to(device)
        labels   = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Loss: {avg_loss:.4f}  Acc: {accuracy:.2f}%"
    )

print("Training Complete")

torch.save(
    model.state_dict(),
    "models/tabular_transformer_40epoch.pth"
)

print("Model Saved")