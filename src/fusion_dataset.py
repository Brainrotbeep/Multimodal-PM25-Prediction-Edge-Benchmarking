import os
import pandas as pd
import torch

from PIL import Image

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torchvision import transforms


class FusionDataset(Dataset):

    def __init__(self, csv_file, image_dir):

        self.df = pd.read_csv(csv_file)

        self.image_dir = image_dir

        self.label_map = {
            "0–50": 0,
            "51–100": 1,
            "101–150": 2,
            "151–200": 3,
            "201–300": 4,
            "301–600": 5
        }

        # station encoding
        self.df["station_id"] = (
            self.df["station_id"]
            .astype("category")
            .cat.codes
        )

        self.features = [
            "station_id",
            "longitude",
            "latitude",
            "year",
            "month",
            "day",
            "hour"
        ]

        # normalize features
        self.scaler = StandardScaler()

        self.df[self.features] = self.scaler.fit_transform(
            self.df[self.features]
        )

        # image transforms
        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(10),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),

            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # IMAGE
        img_path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        # TABULAR
        tabular = torch.tensor(
            row[self.features]
            .astype(float)
            .values,
            dtype=torch.float32
        )

        # LABEL
        label = self.label_map[row["pm25_bin"]]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, tabular, label