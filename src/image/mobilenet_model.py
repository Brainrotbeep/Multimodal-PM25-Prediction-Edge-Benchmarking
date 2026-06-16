import torch.nn as nn
import timm


class MobileNetClassifier(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last stage
        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        return self.backbone(x)