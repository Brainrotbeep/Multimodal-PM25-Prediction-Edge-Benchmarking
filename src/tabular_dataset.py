import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.label_map = {
            "0–50": 0,
            "51–100": 1,
            "101–150": 2,
            "151–200": 3,
            "201–300": 4,
            "301–600": 5
        }

        # convert station_id to numeric
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

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        x = torch.tensor(
            row[self.features]
            .astype(float)
            .values,
            dtype=torch.float32
        )

        y = self.label_map[row["pm25_bin"]]

        y = torch.tensor(y, dtype=torch.long)

        return x, y