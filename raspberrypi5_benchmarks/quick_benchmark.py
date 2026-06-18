import os
import time
import torch
import psutil

from src.fusion.fusion_model import FusionModel

MODEL_PATH = "models/fusion_model_40epoch.pth"

device = torch.device("cpu")

start_temp = os.popen("vcgencmd measure_temp").read().strip()

t0 = time.time()

model = FusionModel()
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)
model.eval()

load_time = time.time() - t0

# Dummy inputs
image = torch.randn(1, 3, 224, 224)
tabular = torch.randn(1, 6)

# Warmup
for _ in range(5):
    with torch.no_grad():
        model(image, tabular)

runs = 50

cpu_before = psutil.cpu_percent(interval=1)

t1 = time.time()

for _ in range(runs):
    with torch.no_grad():
        model(image, tabular)

elapsed = time.time() - t1

cpu_after = psutil.cpu_percent(interval=1)

latency_ms = elapsed / runs * 1000
fps = runs / elapsed

ram_mb = psutil.Process().memory_info().rss / 1024 / 1024

end_temp = os.popen("vcgencmd measure_temp").read().strip()

size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024

print("\n===== BENCHMARK =====")
print("Model:", MODEL_PATH)
print("Size MB:", round(size_mb, 2))
print("Load Time:", round(load_time, 3), "s")
print("Latency:", round(latency_ms, 2), "ms")
print("FPS:", round(fps, 2))
print("RAM MB:", round(ram_mb, 2))
print("CPU Before:", cpu_before)
print("CPU After:", cpu_after)
print("Temp Start:", start_temp)
print("Temp End:", end_temp)
