from fusion_dataset import FusionDataset
from fusion_model import FusionModel


# DATASET
dataset = FusionDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images"
)

print("Dataset Loaded")


# SAMPLE
image, tabular, label = dataset[0]

print("Image Shape:", image.shape)

print("Tabular Shape:", tabular.shape)

print("Label:", label)


# ADD BATCH DIMENSION
image = image.unsqueeze(0)

tabular = tabular.unsqueeze(0)


# MODEL
model = FusionModel()

print("Model Loaded")


# FORWARD PASS
output = model(image, tabular)

print("Output Shape:", output.shape)

print(output)