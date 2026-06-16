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

        # Last block unfreeze
        for param in self.vit.blocks[-1:].parameters():
            param.requires_grad = True

        # PROJECT IMAGE FEATURES
        self.image_projection = nn.Linear(192, 64)

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

        # GATE — BatchNorm added
        self.gate_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),        # gate stable hoga
            nn.Sigmoid()
        )

        # FUSION — deeper + ReLU
        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),                 # non-linearity add ki
            nn.Linear(128, 128)        # extra layer
        )

        self.dropout = nn.Dropout(0.3) # 0.5 se 0.3

        self.classifier = nn.Linear(128, 6)

    def forward(self, image, tabular):

        # IMAGE FEATURES
        image_features = self.vit(image)
        image_features = self.image_projection(image_features)

        # TABULAR FEATURES
        tabular = self.tabular_input(tabular)
        tabular = tabular.unsqueeze(1)
        tabular = self.tabular_transformer(tabular)
        tabular = tabular.squeeze(1)

        # GATE COMPUTATION
        combined = torch.cat([image_features, tabular], dim=1)
        gate = self.gate_layer(combined)

        # GATED FUSION
        fused = (
            gate * image_features
            + (1 - gate) * tabular
        )

        # CLASSIFICATION
        fused = self.fusion_layer(fused)
        fused = self.dropout(fused)
        output = self.classifier(fused)

        return output