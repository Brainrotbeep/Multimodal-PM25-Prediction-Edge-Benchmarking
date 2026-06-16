import torch
import torch.nn as nn
import timm


class FusionMobileNetMLP(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE BRANCH

        self.mobilenet = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=0
        )

        for param in self.mobilenet.parameters():
            param.requires_grad = False

        for param in self.mobilenet.blocks[-2:].parameters():
            param.requires_grad = True

        self.image_projection = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # TABULAR MLP

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

        self.fusion_layer = nn.Sequential(
            nn.Linear(
                64 + 64,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(
            128,
            6
        )

    def forward(self, image, tabular):

        image_features = self.mobilenet(
            image
        )

        image_features = self.image_projection(
            image_features
        )

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

        output = self.classifier(
            fused
        )

        return output