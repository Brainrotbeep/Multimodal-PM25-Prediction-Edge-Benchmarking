import os
import csv
import time
import torch
import psutil
import joblib
import numpy as np

from torchvision import transforms

from src.fusion.fusion_dataset import FusionDataset

from src.fusion.fusion_model import FusionModel
from src.fusion.fusion_model_vit_mlp import FusionViTMLP
from src.fusion.fusion_model_efficientnet import FusionEfficientNet
from src.fusion.fusion_model_efficientnet_mlp import FusionEfficientNetMLP
from src.fusion.fusion_model_mobilenet import FusionMobileNet
from src.fusion.fusion_model_mobilenet_mlp import FusionMobileNetMLP

device = torch.device("cpu")

MODELS = [
    ("ViT_Transformer", FusionModel,
     "models/fusion_model_40epoch.pth"),

    ("ViT_MLP", FusionViTMLP,
     "models/fusion_vit_mlp_40epoch.pth"),

    ("EfficientNet_Transformer",
     FusionEfficientNet,
     "models/fusion_efficientnet_20epoch.pth"),

    ("EfficientNet_MLP",
     FusionEfficientNetMLP,
     "models/fusion_efficientnet_mlp_40epoch.pth"),

    ("MobileNet_Transformer",
     FusionMobileNet,
     "models/fusion_mobilenet_40epoch.pth"),

    ("MobileNet_MLP",
     FusionMobileNetMLP,
     "models/fusion_mobilenet_mlp_40epoch.pth")
]

scaler = joblib.load(
    "models/fusion_scaler.pkl"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = FusionDataset(
    csv_file="data/clean_test_metadata_200.csv",
    image_dir="data/test_images",
    scaler=scaler,
    transform=transform
)

print("Dataset:", len(dataset))

results = []

for name, model_class, model_path in MODELS:

    print("\n" + "="*60)
    print(name)
    print("="*60)

    load_start = time.time()

    model = model_class()

    state = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state)

    model.eval()

    load_time = time.time() - load_start

    warm_img, warm_tab, _ = dataset[0]

    warm_img = warm_img.unsqueeze(0)
    warm_tab = warm_tab.unsqueeze(0)

    with torch.no_grad():
        for _ in range(5):
            model(warm_img, warm_tab)

    latencies = []

    process = psutil.Process()

    benchmark_start = time.time()

    for idx in range(len(dataset)):

        image, tabular, _ = dataset[idx]

        image = image.unsqueeze(0)
        tabular = tabular.unsqueeze(0)

        with torch.no_grad():

            for _ in range(10):

                t1 = time.perf_counter()

                model(image, tabular)

                t2 = time.perf_counter()

                latencies.append(
                    (t2 - t1) * 1000
                )

    runtime = time.time() - benchmark_start

    latency_ms = float(np.mean(latencies))

    fps = 1000.0 / latency_ms

    ram_mb = (
        process.memory_info().rss
        / 1024 / 1024
    )

    size_mb = (
        os.path.getsize(model_path)
        / 1024 / 1024
    )

    results.append([
        name,
        round(size_mb,2),
        round(load_time,2),
        round(latency_ms,2),
        round(fps,2),
        round(ram_mb,2),
        round(runtime,2)
    ])

    print("Latency:", latency_ms)
    print("FPS:", fps)

with open(
    "benchmarks/benchmark_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Size_MB",
        "Load_Time_s",
        "Latency_ms",
        "FPS",
        "RAM_MB",
        "Runtime_s"
    ])

    writer.writerows(results)

print("\nBENCHMARK COMPLETE")