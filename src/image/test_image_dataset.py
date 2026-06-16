print("Starting...")

from image_dataset import ImageDataset

print("Creating dataset...")

dataset = ImageDataset(
    csv_file="data/clean_metadata.csv",
    image_dir="data/images"
)

print("Dataset created")

print("Dataset Size:", len(dataset))

print("Loading first sample...")

image, label = dataset[0]

print("Loaded sample")

print("Image Shape:", image.shape)
print("Label:", label)