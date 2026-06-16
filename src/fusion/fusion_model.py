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

        # FREEZE ViT — fully
        for param in self.vit.parameters():
            param.requires_grad = False

        # Last 1 block unfreeze — thoda fine-tune hoga
        for param in self.vit.blocks[-1:].parameters():
            param.requires_grad = True

        # TABULAR BRANCH
        self.tabular_input = nn.Linear(6, 64)

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
        self.dropout = nn.Dropout(0.5)

        # FINAL CLASSIFIER
        self.classifier = nn.Linear(
            128,
            6
        )

    def forward(self, image, tabular):

        image_features = self.vit(image)

        tabular = self.tabular_input(tabular)
        tabular = tabular.unsqueeze(1)
        tabular = self.tabular_transformer(tabular)
        tabular = tabular.squeeze(1)

        fused = torch.cat([image_features, tabular], dim=1)
        fused = self.fusion_layer(fused)
        fused = self.dropout(fused)

        return self.classifier(fused)