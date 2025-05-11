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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2

# 设置 Matplotlib 支持中文显示和负号正常显示
plt.rcParams['font.family'] = 'SimHei' # 设置字体为黑体 (或其他支持中文的字体)
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 设置 PyTorch Hub 的缓存目录
os.environ['TORCH_HOME'] = './.torch'

# 定义输入、GT 和输出文件夹路径
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1005' # 包含雾图的文件夹
gt_folder = 'data/UAV-M/UAV-benchmark-M/M1005' # 包含对应清晰图的文件夹
max_size = 640 # 模型输入图像的最大边长
result_dir = 'result/dehaze' # 结果保存目录 (图表等)
video_dir = 'result/video' # 生成视频保存目录

# 创建结果和视频目录，如果不存在
os.makedirs(result_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# 定义图像预处理变换：转换为 PyTorch Tensor
transform = transforms.ToTensor()

# 定义要评估的去雾方法列表
# 这些名称应与您的模型配置中的 'method']['dehaze'] 字段对应
dehazes = ['DIP', 'AD_NET', 'AOD_NET']

# 定义模型配置文件的路径
# 假设所有方法使用同一个基础配置，只修改 'method']['dehaze']
yaml_path = 'configs/DIP.yaml'

# 用于存储各方法结果的字典
fps_results = {}
psnr_results = {}
ssim_results = {}

# 获取并排序输入图像文件列表
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) # 过滤常见的图片格式
])

# 遍历每种去雾方法进行评估
for dehaze in dehazes:
    print(f"\n加载去雾模型: {dehaze}")
    # 加载基础配置
    cfg = load_config(yaml_path)
    # 修改配置，指定当前要使用的去雾方法
    cfg['method']['dehaze'] = dehaze
    # 创建模型实例 (假设 create_model 能根据 cfg['method']['dehaze'] 创建对应模型)
    model = create_model(cfg)
    # 加载模型权重 (假设 load_model 方法能根据 cfg 中的信息加载正确的权重)
    model.load_model()
    # 设置设备
    device = cfg['device'] if torch.cuda.is_available() else "cpu"
    model.to(device) # 将模型发送到设备
    model.eval() # 设置模型为评估模式 (关闭 dropout 等)

    # 用于记录当前方法的指标和处理后的帧
    frame_times = [] # 每帧处理时间
    psnr_list = [] # 每帧 PSNR 值
    ssim_list = [] # 每帧 SSIM 值
    processed_frames = [] # 处理后的图像帧 (用于生成视频)

    # 在推理时禁用梯度计算
    with torch.no_grad():
        # 遍历所有图像文件进行处理
        for file_name in image_files:
            image_path = os.path.join(image_folder, file_name) # 雾图完整路径
            gt_path = os.path.join(gt_folder, file_name) # 对应清晰图完整路径

            # 尝试打开图片
            try:
                image = Image.open(image_path).convert("RGB") # 打开雾图并转为 RGB
                gt_image = Image.open(gt_path).convert("RGB") # 打开清晰图并转为 RGB
            except Exception as e:
                print(f"跳过文件: {file_name} ({e})") # 如果打开失败，打印错误并跳过
                continue

            # 将 PIL 图像转换为 PyTorch Tensor (值在 [0, 1])
            image_tensor = transform(image)
            gt_tensor = transform(gt_image)

            # 获取原始尺寸
            orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
            # 计算调整大小的比例因子，确保最长边不超过 max_size
            r = min(1.0, max_size / float(max(orig_w, orig_h)))
            # 计算新的尺寸，确保是 32 的倍数 (模型常见要求)
            new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
            new_w = max(32, int(math.floor(orig_w * r / 32) * 32))
            # 调整雾图和清晰图到模型输入尺寸
            image_resized = F.resize(image_tensor, (new_h, new_w))
            gt_resized = F.resize(gt_tensor, (new_h, new_w))

            # 添加批次维度并发送到设备
            input_tensor = image_resized.unsqueeze(0).to(device)

            # 记录处理开始时间
            start_time = time.time()
            # 使用模型进行预测 (去雾)。假设 model.predict 返回去雾后的图像 Tensor [1, C, H', W']。
            # 注意：如果您的模型结构不同 (例如只输出检测框)，需要调整这里获取去雾图像的方式。
            # 如果模型没有单独的去雾输出，您可能需要修改模型代码或跳过 PSNR/SSIM 计算。
            # 为了与后面的 PSNR/SSIM 计算兼容，假设 model.predict 返回去雾后的图像。
            output = model.predict(input_tensor)

            torch.cuda.synchronize() # 等待 CUDA 操作完成，确保计时准确
            elapsed = time.time() - start_time # 计算处理时间
            frame_times.append(elapsed) # 记录每帧处理时间

            # 移除批次维度，移到 CPU，clamp 值到 [0, 1]
            output_image = output.squeeze(0).cpu().clamp(0, 1)

            # 计算 PSNR 和 SSIM
            # 需要将 Tensor 转换为 NumPy 数组，并调整维度顺序为 (H, W, C)
            # PSNR 和 SSIM 函数期望输入是 NumPy 数组
            psnr_val = psnr(gt_resized.permute(1, 2, 0).numpy(), output_image.permute(1, 2, 0).numpy(), data_range=1)
            ssim_val = ssim(gt_resized.permute(1, 2, 0).numpy(), output_image.permute(1, 2, 0).numpy(), channel_axis=2, data_range=1)

            psnr_list.append(psnr_val) # 记录 PSNR
            ssim_list.append(ssim_val) # 记录 SSIM

            # 将处理后的图像转换为 NumPy 数组 (uint8 格式，值在 0-255) 用于保存和视频生成
            processed_np = (output_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            processed_frames.append(processed_np) # 存储处理后的帧

    # 如果有处理帧 (即 image_files 不为空且没有跳过所有文件)
    if frame_times:
        # 计算平均 FPS, PSNR, SSIM
        fps = len(frame_times) / sum(frame_times)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        # 存储结果
        fps_results[dehaze] = fps
        psnr_results[dehaze] = avg_psnr
        ssim_results[dehaze] = avg_ssim

        # 打印当前方法的评估结果
        print(f"{dehaze} - FPS: {fps:.2f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")

        # 如果有处理后的帧，生成视频
        if processed_frames:
            print(f"正在生成视频: {dehaze}.mp4")
            video_path = os.path.join(video_dir, f'{dehaze}.mp4') # 视频保存路径
            height, width, _ = processed_frames[0].shape # 获取帧尺寸
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 定义视频编码器
            # 创建视频写入对象 (文件名, 编码器, 帧率, 尺寸)
            out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            # 将处理后的帧写入视频文件
            for frame in processed_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 将 RGB 转换为 BGR (OpenCV 视频写入期望 BGR)
                out.write(frame_bgr) # 写入帧

            out.release() # 释放视频写入对象
            print(f"视频已保存到: {video_path}")

    else:
        # 如果没有任何帧被处理
        fps_results[dehaze] = 0
        psnr_results[dehaze] = 0
        ssim_results[dehaze] = 0
        print(f"{dehaze} 未处理图像。")

def plot_metric(result_dict, title, ylabel, filename):
    """
    绘制指标结果的柱状图。

    Args:
        result_dict (dict): 包含指标结果的字典 {方法名: 值}。
        title (str): 图表标题。
        ylabel (str): Y 轴标签。
        filename (str): 保存图表的文件名 (将保存在 result_dir 中)。
    """
    # 按值降序排序结果
    items = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*items) # 分离方法名和值
    plt.figure(figsize=(8, 6)) # 创建图表
    bars = plt.bar(names, values) # 绘制柱状图
    plt.title(title) # 设置标题
    plt.ylabel(ylabel) # 设置 Y 轴标签
    # 设置 Y 轴范围，略大于最大值
    plt.ylim(0, max(values) * 1.2 if values else 1) # 如果 values 为空，设置 ylim 为 (0, 1)
    # 在每个柱状图上方添加数值标签
    for bar in bars:
        h = bar.get_height()
        # 格式化标签，保留两位小数
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02 * max(values) if values else h + 0.02, f"{h:.2f}", ha='center') # 调整文本位置
    plt.tight_layout() # 调整布局，避免标签重叠
    plt.savefig(os.path.join(result_dir, filename)) # 保存图表到文件
    plt.close() # 关闭图表

# 绘制并保存各指标的图表
plot_metric(fps_results, "各去雾方法的平均 FPS", "FPS", "fps.png")
plot_metric(psnr_results, "各去雾方法的平均 PSNR", "PSNR", "psnr.png")
plot_metric(ssim_results, "各去雾方法的平均 SSIM", "SSIM", "ssim.png")
