import os
import csv
import time
import torch
import psutil
import joblib

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
    (
        "ViT_Transformer",
        FusionModel,
        "models/fusion_model_40epoch.pth"
    ),
    (
        "ViT_MLP",
        FusionViTMLP,
        "models/fusion_vit_mlp_40epoch.pth"
    ),
    (
        "EfficientNet_Transformer",
        FusionEfficientNet,
        "models/fusion_efficientnet_20epoch.pth"
    ),
    (
        "EfficientNet_MLP",
        FusionEfficientNetMLP,
        "models/fusion_efficientnet_mlp_40epoch.pth"
    ),
    (
        "MobileNet_Transformer",
        FusionMobileNet,
        "models/fusion_mobilenet_40epoch.pth"
    ),
    (
        "MobileNet_MLP",
        FusionMobileNetMLP,
        "models/fusion_mobilenet_mlp_40epoch.pth"
    )
]


def get_temp():
    try:
        out = os.popen(
            "vcgencmd measure_temp"
        ).read().strip()

        return float(
            out.replace("temp=", "")
               .replace("'C", "")
        )
    except:
        return -1


print("Loading test sample...")

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
    csv_file="data/clean_test_metadata.csv",
    image_dir="data/test_images",
    scaler=scaler,
    transform=transform
)

image, tabular, _ = dataset[0]

image = image.unsqueeze(0)
tabular = tabular.unsqueeze(0)

results = []

for name, model_class, model_path in MODELS:

    print(f"\nBenchmarking {name}")

    temp_start = get_temp()

    load_start = time.time()

    model = model_class()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    load_time = time.time() - load_start

    image_cpu = image
    tabular_cpu = tabular

    with torch.no_grad():
        for _ in range(10):
            _ = model(
                image_cpu,
                tabular_cpu
            )

    runs = 100

    t0 = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(
                image_cpu,
                tabular_cpu
            )

    total_time = time.time() - t0

    latency_ms = (
        total_time / runs
    ) * 1000

    fps = runs / total_time

    ram_mb = (
        psutil.Process()
        .memory_info()
        .rss
        / 1024 / 1024
    )

    temp_end = get_temp()

    temp_rise = (
        temp_end - temp_start
    )

    size_mb = (
        os.path.getsize(model_path)
        / 1024 / 1024
    )

    # Estimated energy
    #
    # Assumption:
    # Pi5 active inference power ≈ 7W

    estimated_energy_j = (
        7.0 * total_time
    )

    results.append([
        name,
        round(size_mb, 2),
        round(load_time, 3),
        round(latency_ms, 2),
        round(fps, 2),
        round(ram_mb, 2),
        round(temp_start, 2),
        round(temp_end, 2),
        round(temp_rise, 2),
        round(estimated_energy_j, 2)
    ])

    print(
        f"Done -> "
        f"{latency_ms:.2f} ms"
    )

os.makedirs(
    "benchmarks",
    exist_ok=True
)

csv_path = (
    "benchmarks/benchmark_results.csv"
)

with open(
    csv_path,
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
        "Temp_Start_C",
        "Temp_End_C",
        "Temp_Rise_C",
        "Estimated_Energy_J"
    ])

    writer.writerows(results)

print(
    "\nBenchmark Complete!"
)

print(
    f"CSV saved to: {csv_path}"
)
