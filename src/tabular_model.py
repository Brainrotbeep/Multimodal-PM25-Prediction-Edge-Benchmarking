import torch
import torch.nn as nn


class TabularTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_layer = nn.Linear(7, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 6)

    def forward(self, x):

        x = self.input_layer(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        x = self.classifier(x)

        return x