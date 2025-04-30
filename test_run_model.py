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
cfg = load_config('configs/exp3.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# 3.模型创建
model = create_model(cfg)
name = cfg['detector']

# 加载模型权重
model.load_checkpoint(f'models/{name}/checkpoints/best_model.pth')
model.eval() # 设置模型为评估模式，关闭 dropout 等

# 将模型移动到 GPU 如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"使用设备: {device}")


# --- 准备用于模拟视频流的单张图片 ---
# 加载图片
image_path = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M0101/img000002.jpg'
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"错误：找不到图片文件 {image_path}")
    exit() # 如果找不到图片，直接退出

# 显示原始图像（可选）
image_np_original = np.array(image)
cv2.imshow("原始图像", cv2.cvtColor(image_np_original, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

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

cv2.imshow("缩放后图像 (模拟输入尺寸)", cv2.cvtColor(img_resized_np, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


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
        result = model(simulated_input_tensor)
        # 注意：这里不进行显示、保存或复杂的后处理，只关注模型的纯推理时间

    # 在计时循环结束后，如果使用 CUDA，同步 GPU，确保所有推理任务完成
    if device.type == 'cuda':
        torch.cuda.synchronize()

end_time = time.perf_counter() # 停止计时

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


# --- 显示最后一帧的推理结果 (可选) ---
# 使用计时阶段最后一帧得到的 'result' 进行显示
# 注意：这里只是为了展示模型输出，不计入 FPS 时间
print("显示最后一帧的推理结果...")

# 检查结果类型，根据你的模型实际输出进行调整
# 假设 result[0] 是一个 tensor，需要进行后处理和可视化
if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], torch.Tensor):
     print("结果[0] 形状/类型:", result[0].shape)

     # 假设 result[0] 是你想要可视化的输出 tensor
     output_tensor = result[0].squeeze() # 移除 batch 维度

     # 确保 output_tensor 是 3 通道以便显示为 RGB 图像
     # 这部分可能需要根据你的模型具体输出格式进行调整
     # 例如，如果它是一个分割掩码或单通道热力图
     if output_tensor.ndim == 2: # 单通道热力图 (H, W)
         output_tensor = output_tensor.unsqueeze(0).repeat(3, 1, 1) # 复制到 3 个通道 (3, H, W)
     elif output_tensor.ndim == 3 and output_tensor.shape[0] == 1: # 单通道 (1, H, W)
          output_tensor = output_tensor.repeat(3, 1, 1) # 复制到 3 个通道 (3, H, W)
     elif output_tensor.ndim != 3 or output_tensor.shape[0] != 3:
         print(f"警告: 输出 tensor 形状 {output_tensor.shape} 不是 3 通道. 无法显示为 RGB 图像.")
         output_tensor = None # 跳过显示

     if output_tensor is not None:
         # 将 tensor 归一化到 [0, 1] 范围以便可视化
         # 添加一个很小的 epsilon 防止 max == min 时除以零
         min_val = output_tensor.min()
         max_val = output_tensor.max()
         # 避免 max_val == min_val 导致除以零
         if max_val == min_val:
              output_tensor = torch.zeros_like(output_tensor) # 如果所有值都一样，显示为黑色或白色，这里设为黑色
         else:
              output_tensor = (output_tensor - min_val) / (max_val - min_val)


         # 将归一化后的 tensor 转换为 NumPy 数组以便 OpenCV 显示
         img_np_output = output_tensor.permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C

         # 将 float [0, 1] 转换为 uint8 [0, 255]
         img_np_output = (img_np_output * 255).clip(0, 255).astype(np.uint8) # 使用 clip 确保值在 0-255 范围内

         # 使用 OpenCV 显示结果
         cv2.imshow("归一化输出 (最后一帧)", cv2.cvtColor(img_np_output, cv2.COLOR_RGB2BGR))
         cv2.waitKey(0) # 等待按键
         cv2.destroyAllWindows() # 关闭窗口
     else:
         print("由于输出 tensor 形状异常，跳过结果显示.")

else:
    print("结果不是预期的格式 (包含 tensor 的 list/tuple). 无法显示输出.")


print("脚本执行完毕.")
