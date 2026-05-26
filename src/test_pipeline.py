import torch

from image_dataset import ImageDataset
from vit_model import ViTClassifier


# DATASET
dataset = ImageDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images"
)

print("Dataset Loaded")


# SAMPLE
image, label = dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)


# ADD BATCH DIMENSION
image = image.unsqueeze(0)

print("Batch Image Shape:", image.shape)


# MODEL
model = ViTClassifier(num_classes=6)

print("Model Loaded")


# FORWARD PASS
output = model(image)

print("Output Shape:", output.shape)

print("Raw Output:", output)