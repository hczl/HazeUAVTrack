import os
import argparse
import cv2

import numpy as np
import torch
from torch import optim

from utils.DataLoader import UAVDataLoaderBuilder
from utils.common import create_model, create_data
from utils.config import load_config



# 1.导入设置
cfg = load_config('configs/exp2.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# 3.模型创建
model = create_model(cfg)
# 4.模型训练
# model.train_model(train_loader=train_loader, val_loader=val_loader, clean_loader=clean_loader, num_epochs=cfg['train']['epochs'])
# model.load_state_dict(torch.load('models/IA_YOLOV3/checkpoints/checkpoint_epoch_80.pth'))
model.load_checkpoint('models/DE_NET/checkpoints/checkpoint_epoch_30.pth')
model.enhancement.eval()
model.yolov3.eval()
# Load the image
image = cv2.imread('data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M0101/img000002.jpg')

# 检查图片是否成功加载
if image is None:
    print("错误：无法加载图片。请检查路径是否正确。")
    exit()  # 或者进行其他错误处理

# --- 图像缩放处理 ---
# 获取原始图像尺寸
h_orig, w_orig = image.shape[:2]

target_size = 1024

# 计算缩放比例，基于长边
scale = target_size / max(h_orig, w_orig)

# 计算按比例缩放后的中间尺寸
h_inter = int(h_orig * scale)
w_inter = int(w_orig * scale)

h_new = ((h_inter + 31) // 32) * 32
w_new = ((w_inter + 31) // 32) * 32

resized_image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
# --- 缩放处理结束 ---

# 继续你的后续处理步骤，使用 resized_image
input_image1 = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# 转换为 float 并归一化到 [0, 1]
input_image = input_image1.astype(np.float32) / 255.0

input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
# 模型推理
with torch.no_grad():
    results = model.predict(input_image)
# 假设 results 是 model.predict(source) 的输出

# 检查 results 是否是列表 (通常是)
if isinstance(results, list):
    # 遍历列表中的每一个结果对象
    for i, result in enumerate(results):
        # 现在 result 是一个单独的 ultralytics.engine.results.Results 对象
        # 它拥有 plot() 方法
        boxes = result.boxes
        print(boxes)
        # 现在你可以处理这个 annotated_image (NumPy 数组)
        # 例如，使用 OpenCV 显示或保存
        import cv2

        # x, y, w, h = int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])

        # # 获取目标ID作为标签
        # label = f"ID: {row['target_id']}"
        #
        # # 绘制边界框
        # cv2.rectangle(input_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 使用绿色框
        #
        # # 在左上角添加标签
        # cv2.putText(input_image1, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存图像 (可选)
        output_filename = f"image_with_boxes_{i}.jpg"
        cv2.imwrite(output_filename, input_image1)
        print(f"Saved annotated image to {output_filename}")

        # 或者使用 result 对象自带的 save() 方法保存
        # 它会保存到 results.save_dir 目录下
        result.save()
        print(f"Saved annotated image for item {i} to {result.save_dir}")

else:
    # 如果 predict 返回的是单个 Results 对象 (较少见，但为代码健壮性考虑)
    # 虽然根据错误，你的情况不是这样
    annotated_image = results.plot()
    # 处理这个单独的 annotated_image
    # ... 显示或保存 ...
    results.save()
    print(f"Saved single annotated image to {results.save_dir}")

