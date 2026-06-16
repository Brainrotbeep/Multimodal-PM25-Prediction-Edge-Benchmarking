import os
import pandas as pd
import torch

from PIL import Image

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torchvision import transforms


class FusionDataset(Dataset):

    def __init__(self, csv_file, image_dir, scaler=None, transform=None):

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

        # station_id hata diya — alag datasets mein alag codes bante hain
        self.features = [
            "longitude",
            "latitude",
            "year",
            "month",
            "day",
            "hour"
        ]

        # scaler bahar se pass karo test ke liye
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[self.features] = self.scaler.fit_transform(
                self.df[self.features]
            )
        else:
            self.scaler = scaler
            self.df[self.features] = self.scaler.transform(  # sirf transform, fit nahi
                self.df[self.features]
            )

        # transform bahar se pass karo — train aur test alag hoga
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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
        label = torch.tensor(label, dtype=torch.long)

        return image, tabular, label