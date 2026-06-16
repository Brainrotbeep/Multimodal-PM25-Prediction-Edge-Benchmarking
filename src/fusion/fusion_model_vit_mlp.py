import torch
import torch.nn as nn
import timm


class FusionViTMLP(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE BRANCH

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.blocks[-1:].parameters():
            param.requires_grad = True

        # TABULAR MLP BRANCH

        self.tabular_mlp = nn.Sequential(

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

        # FUSION

        self.fusion_layer = nn.Linear(
            192 + 64,
            128
        )

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(
            128,
            6
        )

    def forward(self, image, tabular):

        image_features = self.vit(image)

        tabular_features = self.tabular_mlp(
            tabular
        )

        fused = torch.cat(
            [
                image_features,
                tabular_features
            ],
            dim=1
        )

        fused = self.fusion_layer(
            fused
        )

        fused = self.dropout(
            fused
        )

        return self.classifier(
            fused
        )