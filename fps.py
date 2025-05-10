import os
import time
import math
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.config import load_config
from utils.create import create_model
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ---- 初始设置 ----
os.environ['TORCH_HOME'] = './.torch'
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1101'
transform = transforms.Compose([transforms.ToTensor()])
max_size = 640

# ---- 去雾方法和检测器 ----
yaml_paths = ['configs/DRIFT_NET.yaml']
fps_results = {}

# ---- 获取图像列表 ----
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# ---- 对每组组合计算 FPS ----
for yaml_path in yaml_paths:
    print(f"\n加载配置: {yaml_path}")
    cfg = load_config(yaml_path)

    model = create_model(cfg)
    model.load_model()
    device = cfg['device']
    model.to(device)
    print(f"使用设备: {device}")

    frame_times = []

    with torch.no_grad():
        for image_path in image_files:
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                continue

            image_tensor = transform(image)
            orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
            r = min(1.0, max_size / float(max(orig_w, orig_h)))
            new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
            new_w = max(32, int(math.floor(orig_w * r / 32) * 32))
            image_resized = F.resize(image_tensor, (new_h, new_w))
            input_tensor = image_resized.unsqueeze(0).to(device)

            start_time = time.time()
            _ = model.predict(input_tensor)
            torch.cuda.synchronize()  # 保证计时准确
            elapsed = time.time() - start_time
            frame_times.append(elapsed)

    # ---- FPS 统计 ----
    if frame_times:
        avg_fps = len(frame_times) / sum(frame_times)
        fps_results[yaml_path] = avg_fps
        print(f"{yaml_path} 平均 FPS：{avg_fps:.2f}")
    else:
        fps_results[yaml_path] = 0
        print(f"{yaml_path} 未处理图像。")

# ---- 绘制 FPS 对比图 ----
sorted_results = sorted(fps_results.items(), key=lambda x: x[1], reverse=True)
names, fps_values = zip(*sorted_results)

plt.figure(figsize=(10, 6))
bars = plt.bar(names, fps_values)
plt.title("不同去雾+检测器组合的平均 FPS 对比")
plt.xlabel("组合 (去雾_检测器)")
plt.ylabel("平均 FPS")
plt.ylim(0, max(fps_values) * 1.2)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}", ha='center')
plt.tight_layout()
plt.savefig("fps_comparison_all_combinations.png")
plt.show()
