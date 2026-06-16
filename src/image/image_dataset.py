import os
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):  # transform=None added

        self.df = pd.read_csv(csv_file)

        self.image_dir = image_dir

        # Agar bahar se transform pass kiya toh wo use karo
        # warna default use hoga (eval/inference ke liye fallback)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.label_map = {
            "0–50": 0,
            "51–100": 1,
            "101–150": 2,
            "151–200": 3,
            "201–300": 4,
            "301–600": 5
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        label = self.label_map[row["pm25_bin"]]

        return image, label