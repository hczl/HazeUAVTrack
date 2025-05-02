import math
import os
import argparse
import cv2
import time  # 导入时间模块

import numpy as np
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as F

# 假设这些导入是正确的，基于你的代码结构
# from utils.DataLoader import UAVDataLoaderBuilder # 在这个脚本中不需要数据加载器
# from utils.common import create_model, create_data # create_data也不需要
from utils.common import create_model # 只保留create_model
from utils.config import load_config

# 设置 TORCH_HOME 环境变量（如果需要）
os.environ['TORCH_HOME'] = './.torch'

# 1.导入设置
cfg = load_config('configs/exp1.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# 3.模型创建
model = create_model(cfg)
name = cfg['detector']

# 加载模型权重
model.load_checkpoint(f'models/{name}/checkpoints/best_model.pth')

# 将模型移动到 GPU 如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"使用设备: {device}")


# --- 准备用于模拟视频流的单张图片 ---
# 加载图片
image_path = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1101/img000002.jpg'
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"错误：找不到图片文件 {image_path}")
    exit() # 如果找不到图片，直接退出

# 显示原始图像（可选）
image_np_original = np.array(image)
# cv2.imshow("原始图像", cv2.cvtColor(image_np_original, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 将 PIL 图片转换为 Tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image) # 重命名变量以避免与 numpy 数组混淆


# --- 图像缩放处理（用于模拟输入尺寸） ---
# 获取原始图片尺寸
original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

# 定义最大尺寸，例如 608
max_size = 608 # 很多 YOLO 模型常用尺寸，且是 32 的倍数

# 计算缩放比例，保持长宽比
r = min(1.0, max_size / float(original_w))
r = min(r, max_size / float(original_h))

# 根据比例计算新的尺寸
new_h = int(round(original_h * r))
new_w = int(round(original_w * r))

# 确保新尺寸是 32 的倍数
new_h = max(32, int(math.floor(new_h / 32) * 32)) # 确保最小尺寸是 32x32
new_w = max(32, int(math.floor(new_w / 32) * 32))

# 缩放图片 Tensor
img_resized_tensor = F.resize(image_tensor, (new_h, new_w))

# 将缩放后的 Tensor 转换为 NumPy 格式以便显示（可选）
img_resized_np = img_resized_tensor.permute(1, 2, 0).cpu().numpy()
if img_resized_np.dtype != np.uint8:
    # 将 float [0, 1] 缩放到 uint8 [0, 255]
    img_resized_np = (img_resized_np * 255).clip(0, 255).astype(np.uint8)

# cv2.imshow("缩放后图像 (模拟输入尺寸)", cv2.cvtColor(img_resized_np, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 将缩放后的 Tensor 转换为 torch tensor，添加 batch 维度，移动到设备
# 缩放后的 tensor 已经是 (C, H, W) 格式，只需添加 batch 维度并移动
# 这个 tensor 将作为模拟视频流每一帧的输入
simulated_input_tensor = img_resized_tensor.unsqueeze(0).to(device)


# --- 模拟视频流推理并计算 FPS ---
print("\n--- 开始模拟视频流 FPS 测试 ---")

# 定义模拟参数
num_warmup_frames = 10  # 预热帧数，让模型和设备进入工作状态
num_timed_frames = 300  # 用于计算 FPS 的帧数，越多结果越稳定

print(f"模拟输入尺寸: ({new_w}, {new_h})")

# 预热阶段
print(f"执行预热 ({num_warmup_frames} 帧)...")
with torch.no_grad(): # 在推理时禁用梯度计算
    for _ in range(num_warmup_frames):
        # 使用同一张预处理过的图片进行模拟推理
        _ = model(simulated_input_tensor)
        # 如果使用 CUDA，同步 GPU，确保前向传播完成
        if device.type == 'cuda':
             torch.cuda.synchronize()
print("预热完成.")

# 计时阶段
print(f"执行计时推理 ({num_timed_frames} 帧)...")
start_time = time.perf_counter() # 使用 perf_counter 进行高精度计时

with torch.no_grad(): # 在推理时禁用梯度计算
    for i in range(num_timed_frames):
        # 使用同一张预处理过的图片进行模拟推理
        result = model.predict(simulated_input_tensor)
        # 注意：这里不进行显示、保存或复杂的后处理，只关注模型的纯推理时间

    # 在计时循环结束后，如果使用 CUDA，同步 GPU，确保所有推理任务完成
    if device.type == 'cuda':
        torch.cuda.synchronize()

end_time = time.perf_counter() # 停止计时
# print(result.shape)
# 计算并打印 FPS
total_inference_duration = end_time - start_time
# 避免除以零
if total_inference_duration > 0:
    fps = num_timed_frames / total_inference_duration
else:
    fps = float('inf') # 如果时间为零，FPS 无穷大

print("\n--- FPS 测试结果 (模拟视频流) ---")
print(f"处理帧数: {num_timed_frames}")
print(f"总推理时间: {total_inference_duration:.4f} 秒")
print(f"估计 FPS: {fps:.2f}")
print("------------------------------------\n")
print(result[0])

# --- 显示最后一帧的推理结果 (可选) ---
# 使用计时阶段最后一帧得到的 'result' 进行显示
# 注意：这里只是为了展示模型输出，不计入 FPS 时间
print("显示最后一帧的推理结果...")

print("显示最后一帧的推理结果...")

# 假设 result[0] 是 numpy.ndarray，形状为 (N, 5) 或 (N, 6)  [x1, y1, x2, y2, score, (cls)]
if isinstance(result[0], np.ndarray) and result[0].ndim == 2 and result[0].shape[1] >= 5:
    boxes = result[0]

    # 复制原始图像，用于绘图（BGR 格式）
    img_draw = cv2.cvtColor(img_resized_np.copy(), cv2.COLOR_RGB2BGR)

    for box in boxes:
        x1, y1, x2, y2, conf = box[:5]
        # 转为整数
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{conf:.2f}"
        # 绘制框
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # 写置信度
        cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("检测结果", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("输出不是检测框格式 (N,5)，跳过可视化。")
