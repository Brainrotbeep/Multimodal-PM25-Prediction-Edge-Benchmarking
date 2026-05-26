import torch

from tabular_model import TabularTransformer


model = TabularTransformer()

x = torch.randn(1, 7)

output = model(x)

print("Output Shape:", output.shape)

print(output)