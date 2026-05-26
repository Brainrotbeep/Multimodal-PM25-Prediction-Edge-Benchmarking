import torch
import torch.nn as nn

import timm


class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE BRANCH
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        # FREEZE ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # TABULAR BRANCH
        self.tabular_input = nn.Linear(7, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.tabular_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # FUSION
        self.fusion_layer = nn.Linear(
            192 + 64,
            128
        )

        # DROPOUT
        self.dropout = nn.Dropout(0.3)

        # FINAL CLASSIFIER
        self.classifier = nn.Linear(
            128,
            6
        )

    def forward(self, image, tabular):

        # IMAGE FEATURES
        image_features = self.vit(image)

        # TABULAR FEATURES
        tabular = self.tabular_input(tabular)

        tabular = tabular.unsqueeze(1)

        tabular = self.tabular_transformer(tabular)

        tabular = tabular.squeeze(1)

        # CONCATENATE
        fused = torch.cat(
            [image_features, tabular],
            dim=1
        )

        # FUSION
        fused = self.fusion_layer(fused)

        # DROPOUT
        fused = self.dropout(fused)

        # FINAL CLASSIFICATION
        output = self.classifier(fused)

        return output