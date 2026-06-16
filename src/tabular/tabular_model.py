import torch
import torch.nn as nn


class TabularTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_layer = nn.Linear(6, 64)

        self.norm = nn.LayerNorm(64)          # stabilizes transformer input

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True,
            dropout=0.1                        # transformer internal dropout
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3                       # 2 se 3
        )

        self.dropout = nn.Dropout(0.3)         # overfitting rokne ke liye

        self.classifier = nn.Linear(64, 6)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.norm(x)                       # normalize before transformer
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x