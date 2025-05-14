import os
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
# Assumed to be available
from utils.config import load_config
from utils.create import create_model
# Assumed HazeUAVTrack class is defined elsewhere and importable
# from HazeUAVTrack import HazeUAVTrack

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import traceback # 导入 traceback 用于打印详细错误

# 设置 Matplotlib 支持中文显示和负号正常显示
plt.rcParams['font.family'] = 'SimHei' # 设置字体为黑体 (或其他支持中文的字体)
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 设置 PyTorch Hub 的缓存目录
os.environ['TORCH_HOME'] = './.torch'

# 定义输入、GT 和输出文件夹路径
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog_050/M1005' # 包含雾图的文件夹
gt_folder = 'data/UAV-M/UAV-benchmark-M/M1005' # 包含对应清晰图的文件夹
max_size = 1024 # 模型输入图像的最大边长
result_dir = 'result/dehaze' # 结果保存目录 (图表等)
video_dir = 'result/video' # 生成视频保存目录

# 创建结果和视频目录，如果不存在
os.makedirs(result_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# 定义图像预处理变换：转换为 PyTorch Tensor
# transforms.ToTensor() 将 PIL Image (HWC, uint8, [0, 255]) 转换为 Tensor (CHW, float, [0.0, 1.0])
transform_to_tensor = transforms.ToTensor()


# Define the mean and std used during training normalization
# These should match the values used in your create_data function
# Make them Tensors with shape [1, C, 1, 1] for easy broadcasting with [1, C, H, W] output
# Assuming RGB channels (3)
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# 定义要评估的去雾方法列表
# 这些名称应与您的模型配置中的 'method']['dehaze'] 字段对应
dehazes = ['AD_NET','DIP', 'FALCON', 'AOD_NET', 'FFA']

# 定义模型配置文件的路径
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

    # --- Create and load model ---
    # Assuming create_model function is available and returns an instance of HazeUAVTrack or similar
    # Assuming model has .to(device), .eval(), .load_model(), and .predict() methods
    model = create_model(cfg)

    # Set device
    device = cfg['device'] if torch.cuda.is_available() else "cpu"
    model.to(device) # Move model to device

    # Load model weights
    try:
        # Assuming model.load_model() loads the state_dict correctly for inference
        model.load_model() # Ensure this method correctly loads weights for inference
    except Exception as e:
        print(f"警告: 加载模型权重失败 for {dehaze}: {e}")
        print("将使用未加载权重的模型。请检查 load_model 方法。")
        traceback.print_exc() # 打印详细错误信息

    # Set model to evaluation mode (disables dropout, batchnorm tracking stats etc.)
    model.eval()

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
                image_pil = Image.open(image_path).convert("RGB") # 打开雾图并转为 RGB
                gt_image_pil = Image.open(gt_path).convert("RGB") # 打开清晰图并转为 RGB
            except Exception as e:
                print(f"跳过文件: {file_name} ({e})") # 如果打开失败，打印错误并跳过
                continue

            # 将 PIL 图像转换为 PyTorch Tensor (值在 [0, 1]，RGB, [C, H, W])
            image_tensor = transform_to_tensor(image_pil)
            gt_tensor = transform_to_tensor(gt_image_pil) # GT 也需要 Tensor 形式用于 resize 和 PSNR/SSIM 计算

            # 获取原始尺寸
            orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
            # 计算调整大小的比例因子，确保最长边不超过 max_size
            r = min(1.0, max_size / float(max(orig_w, orig_h)))
            # 计算新的尺寸，确保是 32 的倍数 (模型常见要求)
            new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
            new_w = max(32, int(math.floor(orig_w * r / 32) * 32))
            # 调整雾图和清晰图到模型输入尺寸
            image_resized = F.resize(image_tensor, (new_h, new_w))
            gt_resized = F.resize(gt_tensor, (new_h, new_w)) # GT 也需要 resize 到相同尺寸进行对比

            # 添加批次维度并发送到设备
            # Apply normalization using the same mean/std as training
            # Assuming your model expects normalized input like in training
            normalized_input_tensor = F.normalize(image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensor = normalized_input_tensor.unsqueeze(0).to(device)


            # 记录处理开始时间
            start_time = time.time()
            # Use the model to predict (dehaze). Assume model.predict returns a dehazed image Tensor [1, C, H', W'].
            # This output Tensor is expected to be in the standardized space.
            output = model.predict(input_tensor) # Assume output is a Tensor [1, C, H', W'] (STANDARDIZED)

            # Ensure output is a tensor before processing
            if not isinstance(output, torch.Tensor):
                 print(f"警告: 模型预测输出不是 Tensor for {file_name}. Skipping.")
                 continue # Skip this file if output is not a Tensor

            torch.cuda.synchronize() # Wait for CUDA operations to complete for accurate timing
            elapsed = time.time() - start_time # Calculate processing time
            frame_times.append(elapsed) # Record processing time per frame

            # --- Process model output for metrics and visualization ---

            # Assuming model output is a standardized Tensor [1, C, H, W]
            # Needs denormalization and clamping to [0, 1] range
            # Move NORM_MEAN and NORM_STD to the same device as output
            output_denorm = output * NORM_STD.to(output.device) + NORM_MEAN.to(output.device)

            # Clamp values to [0, 1] after denormalization
            output_clamped = torch.clamp(output_denorm, 0, 1)

            # Remove batch dimension and move to CPU for PSNR/SSIM and saving
            output_image_cpu = output_clamped.squeeze(0).cpu() # Shape [C, H, W] (float, [0, 1])


            # Calculate PSNR and SSIM
            # Needs conversion to NumPy array with dimension order (H, W, C)
            # PSNR and SSIM functions expect NumPy arrays with values in [0, 1] or [0, 255]
            # GT image gt_resized is Tensor [C, H, W], float, [0, 1]
            # Processed image output_image_cpu is Tensor [C, H, W], float, [0, 1] (after denorm and clamp)
            try:
                # Convert to NumPy (H, W, C)
                gt_np = gt_resized.permute(1, 2, 0).numpy()
                output_np_float = output_image_cpu.permute(1, 2, 0).numpy()

                psnr_val = psnr(gt_np, output_np_float, data_range=1)
                # SSIM needs channel_axis parameter
                ssim_val = ssim(gt_np, output_np_float, channel_axis=2, data_range=1)

                psnr_list.append(psnr_val) # Record PSNR
                ssim_list.append(ssim_val) # Record SSIM
            except Exception as e:
                 print(f"警告: 计算 PSNR/SSIM 失败 for {file_name}: {e}")
                 traceback.print_exc()
                 psnr_list.append(np.nan) # Record NaN indicating calculation failure
                 ssim_list.append(np.nan) # Record NaN indicating calculation failure


            # Convert processed image to NumPy array (uint8 format, values in 0-255) for saving and video generation
            # output_image_cpu is [C, H, W], float, [0, 1]
            # Permute to [H, W, C], multiply by 255, convert to uint8
            processed_np = (output_image_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            processed_frames.append(processed_np) # Store processed frame

    # If any frames were processed (i.e., image_files was not empty and no files were skipped)
    # Filter out NaN values from PSNR/SSIM lists before calculating mean
    valid_psnr = [x for x in psnr_list if not math.isnan(x)]
    valid_ssim = [x for x in ssim_list if not math.isnan(x)]

    if frame_times: # Check if any frames were processed successfully for timing
        # Calculate average FPS
        fps = len(frame_times) / sum(frame_times)

        # Calculate average PSNR, SSIM (only for valid values)
        avg_psnr = np.mean(valid_psnr) if valid_psnr else 0 # Avoid calculating mean of empty list
        avg_ssim = np.mean(valid_ssim) if valid_ssim else 0 # Avoid calculating mean of empty list


        # Store results
        fps_results[dehaze] = fps
        psnr_results[dehaze] = avg_psnr
        ssim_results[dehaze] = avg_ssim

        # Print current method's evaluation results
        print(f"{dehaze} - FPS: {fps:.2f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")

        # If there are processed frames, generate video
        if processed_frames:
            print(f"正在生成视频: {dehaze}.mp4")
            video_path = os.path.join(video_dir, f'{dehaze}.mp4') # Video save path
            height, width, _ = processed_frames[0].shape # Get frame dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define video codec (for mp4)
            # Ensure video dimensions are even, otherwise VideoWriter might fail
            width = width if width % 2 == 0 else width - (width % 2)
            height = height if height % 2 == 0 else height - (height % 2)

            try:
                 out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

                 # Write processed frames to video file
                 for frame in processed_frames:
                     # Resize frame to the final video size if necessary (due to potential odd dimensions from original resize)
                     if frame.shape[0] != height or frame.shape[1] != width:
                          frame = cv2.resize(frame, (width, height))

                     # Processed frames are currently RGB uint8 [0, 255] HWC
                     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert RGB to BGR (OpenCV video writer expects BGR)
                     out.write(frame_bgr) # Write frame

                 out.release() # Release video writer object
                 print(f"视频已保存到: {video_path}")
            except Exception as e:
                 print(f"错误: 生成视频 {video_path} 失败: {e}")
                 traceback.print_exc()


    else:
        # If no frames were processed
        fps_results[dehaze] = 0
        psnr_results[dehaze] = 0
        ssim_results[dehaze] = 0
        print(f"{dehaze} 未处理图像。")


def plot_metric(result_dict, title, ylabel, filename, is_integer=False, reverse_sort=True):
    if not result_dict:
        print(f"没有数据可绘制 {title}。")
        return

    sorted_items = sorted(result_dict.items(),
                          key=lambda item: item[1] if not (isinstance(item[1], float) and math.isnan(item[1])) else (-float('inf') if reverse_sort else float('inf')),
                          reverse=reverse_sort)

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)

    # 设置x轴标签倾斜
    plt.xticks(rotation=45, ha='right')  # 这里设置标签斜着显示

    valid_values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    max_val = max(valid_values) if valid_values else 0.1
    min_val = min(valid_values) if valid_values else 0

    if 'MOTA' in title:
        plt.ylim(min(0, min_val * 1.1 if min_val < 0 else 0), max(0.1, max_val * 1.2))
    elif 'ID切换' in title:
        plt.ylim(0, max(1, max_val * 1.2))
    else:
        lower_bound = min(0, min_val * (1.1 if min_val < 0 else 0.9)) if valid_values else 0
        upper_bound = max(0.1, max_val * 1.2) if valid_values else 1.0
        plt.ylim(lower_bound, upper_bound)

    for bar in bars:
        h = bar.get_height()
        if math.isnan(h):
            label_text = "NaN"
        elif is_integer:
            label_text = f"{int(h)}"
        else:
            label_text = f"{h:.2f}" if abs(h) >= 0.01 else f"{h:.4f}"

        y_range = plt.ylim()[1] - plt.ylim()[0]
        vertical_offset = y_range * 0.02 if y_range > 0 else 0.02

        if h >= 0:
            plt.text(bar.get_x() + bar.get_width()/2, h + vertical_offset, label_text, ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, h - vertical_offset, label_text, ha='center', va='top')

    plt.tight_layout()
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存到: {save_path}")


# Plot and save charts for each metric
plot_metric(fps_results, "各去雾方法的平均 FPS", "FPS", "fps.png")
plot_metric(psnr_results, "各去雾方法的平均 PSNR", "PSNR", "psnr.png") # Corrected filename
plot_metric(ssim_results, "各去雾方法的平均 SSIM", "SSIM", "ssim.png")

print("\n评估完成。图表已保存。")
