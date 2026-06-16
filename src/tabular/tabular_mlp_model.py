import torch
import torch.nn as nn


class TabularMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)

        )

        self.classifier = nn.Linear(
            64,
            6
        )

    def forward(self, x):

        features = self.network(x)

        output = self.classifier(
            features
        )

        return output