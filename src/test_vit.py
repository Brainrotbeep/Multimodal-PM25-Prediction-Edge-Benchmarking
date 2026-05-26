import torch

from vit_model import ViTClassifier

model = ViTClassifier(num_classes=6)

print(model)

x = torch.randn(1, 3, 224, 224)

output = model(x)

print("Output Shape:", output.shape)