import torch
import torch.nn as nn
import timm


class FusionEfficientNetAttention(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE BRANCH

        self.efficientnet = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        for param in self.efficientnet.parameters():
            param.requires_grad = False

        self.image_projection = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # TABULAR BRANCH

        self.tabular_input = nn.Linear(6, 64)

        self.tabular_norm = nn.LayerNorm(64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True,
            dropout=0.1
        )

        self.tabular_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # ATTENTION FUSION

        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(
            128,
            6
        )

    def forward(self, image, tabular):

        # IMAGE FEATURES

        image_features = self.efficientnet(image)

        image_features = self.image_projection(
            image_features
        )

        # TABULAR FEATURES

        tabular = self.tabular_input(tabular)

        tabular = self.tabular_norm(tabular)

        tabular = tabular.unsqueeze(1)

        tabular = self.tabular_transformer(
            tabular
        )

        tabular = tabular.squeeze(1)

        # ATTENTION

        combined = torch.cat(
            [
                image_features,
                tabular
            ],
            dim=1
        )

        weights = self.attention(
            combined
        )

        weights = torch.softmax(
            weights,
            dim=1
        )

        image_weight = weights[:, 0].unsqueeze(1)

        tabular_weight = weights[:, 1].unsqueeze(1)

        fused = (
            image_weight * image_features
            +
            tabular_weight * tabular
        )

        fused = self.fusion_layer(
            fused
        )

        output = self.classifier(
            fused
        )

        return output