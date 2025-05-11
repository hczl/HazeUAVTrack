import torch
import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_image_for_depth(args):
    """处理单张图片以计算深度图"""
    img_name, img_path, transform, midas, device, current_depth_path = args
    img_file = img_path / img_name
    image = cv2.imread(str(img_file)) # 读取图片
    if image is None:
        return img_name, None # 读取失败则返回 None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为 RGB 格式
    input_tensor = transform(image_rgb).to(device) # 应用 MiDaS 预处理并发送到设备

    with torch.no_grad():
        prediction = midas(input_tensor) # 使用 MiDaS 模型预测深度
    depth_map = prediction.squeeze().cpu().numpy() # 获取深度图，移除批次维度，移到 CPU，转为 numpy
    # 归一化深度图到 0-255 范围，并转为 uint8
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 将深度图调整回原图尺寸
    resized_depth = cv2.resize(depth_map_norm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(current_depth_path / img_name), resized_depth) # 保存深度图
    return img_name, resized_depth # 返回文件名和深度图

def process_image_for_fog(args):
    """处理单张图片以生成雾图"""
    img_name, img_path, depth, fog_color, fog_strength, current_hazy_path = args
    img_file = img_path / img_name
    image = cv2.imread(str(img_file)) # 读取原始图片
    if image is None or depth is None:
        return # 如果图片或深度图无效则返回
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为 RGB 格式
    depth_norm = depth.astype(np.float32) / 255 # 归一化深度图到 0-1
    # 计算雾强度，与深度图归一化值呈负相关
    fog_intensity = np.clip(1 - depth_norm, 0, 1) * fog_strength
    # 创建雾层，颜色由 fog_color 指定
    fog_layer = np.ones_like(image, dtype=np.float32) * fog_color
    # 根据大气散射模型生成雾图
    foggy = image * (1 - fog_intensity[:, :, None]) + fog_layer * fog_intensity[:, :, None]
    foggy = np.clip(foggy, 0, 255).astype(np.uint8) # 确保像素值在 0-255 范围内并转为 uint8
    output_file = current_hazy_path / img_name # 生成雾图的输出路径
    cv2.imwrite(str(output_file), cv2.cvtColor(foggy, cv2.COLOR_RGB2BGR)) # 保存雾图

def MiDaS_Deep(input_path, fog_strength, num_worker=8):
    """
    使用 MiDaS 估计深度并生成雾图。

    Args:
        input_path (str): 输入图像数据集的根目录。
        fog_strength (float): 雾的强度，介于 0 和 1 之间。
        num_worker (int): 用于并行处理的线程数。

    Returns:
        str: 生成的雾图数据集的根目录路径。
    """
    input_path = Path(input_path) # 转换为 Path 对象
    dataset_name = input_path.name # 数据集名称
    fog_strength_str = f"fog_{int(fog_strength * 100):03d}" # 根据雾强度生成输出目录名后缀
    # 构建输出路径
    base_output = input_path.parent / Path(f"MiDaS_Deep_{dataset_name}_{fog_strength_str}")
    depth_path = base_output / 'depth_temp' # 临时深度图存储路径
    if depth_path.exists():
        shutil.rmtree(depth_path) # 如果临时深度目录存在，先删除
    if os.path.exists(base_output):
        print(f"目标目录已存在，跳过生成：{base_output.resolve()}")
        return str(base_output.resolve()) # 如果输出目录已存在，则直接返回

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 选择设备
    model_type = 'DPT_Large' # MiDaS 模型类型
    torch.hub.set_dir('/path/to/custom/cache')  # 可自定义 PyTorch Hub 的缓存目录
    midas = torch.hub.load("intel-isl/MiDaS", model_type) # 加载 MiDaS 模型
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms") # 加载 MiDaS 预处理变换
    midas.to(device) # 将模型发送到设备
    midas.eval() # 设置模型为评估模式

    fog_color = np.array([200, 200, 200], dtype=np.uint8) # 雾的颜色 (这里是灰白色)
    subfolders = [f for f in os.listdir(input_path) if (input_path / f).is_dir()] # 查找输入目录下的所有子文件夹 (通常代表序列)

    for subfolder in subfolders: # 遍历每个子文件夹
        img_path = input_path / subfolder # 当前图片所在的子文件夹路径
        current_depth_path = depth_path / subfolder # 当前子文件夹对应的深度图临时存储路径
        current_hazy_path = base_output / subfolder # 当前子文件夹对应的雾图输出路径
        os.makedirs(current_depth_path, exist_ok=True) # 创建深度图临时存储目录
        os.makedirs(current_hazy_path, exist_ok=True) # 创建雾图输出目录

        imglist = sorted(os.listdir(img_path)) # 获取子文件夹中的图片列表并排序
        # 根据模型类型选择对应的预处理变换
        transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

        # 并行计算深度图
        depth_maps = [None] * len(imglist) # 存储原始深度图的列表
        # 为每张图片准备处理参数
        args = [(img_name, img_path, transform, midas, device, current_depth_path) for img_name in imglist]
        with ThreadPoolExecutor(max_workers=num_worker) as executor: # 使用线程池并行执行
            # 遍历执行结果，填充 depth_maps 列表
            for result in tqdm(executor.map(process_image_for_depth, args), total=len(args), desc=f'{subfolder} - 深度图'):
                img_name, depth = result
                if depth is not None:
                    idx = imglist.index(img_name)
                    depth_maps[idx] = depth

        # 滑动平均平滑深度图
        smoothed_depth_maps = [] # 存储平滑后深度图的列表
        window_size = 5 # 平滑窗口大小
        for i in range(len(depth_maps)): # 遍历每张图片的深度图
            if depth_maps[i] is None:
                smoothed_depth_maps.append(None)
                continue
            start = max(0, i - window_size // 2) # 计算窗口起始索引
            end = min(len(depth_maps), i + window_size // 2 + 1) # 计算窗口结束索引
            # 获取窗口内的有效深度图
            window = [d for d in depth_maps[start:end] if d is not None]
            if window: # 如果窗口内有有效深度图
                smoothed = np.mean(window, axis=0) # 计算平均值进行平滑
                smoothed_depth_maps.append(smoothed.astype(np.uint8)) # 添加平滑后的深度图 (uint8)
            else:
                 smoothed_depth_maps.append(None) # 如果窗口内没有有效深度图，添加 None


        # 并行生成雾图
        fog_args = [
            # 为每张图片准备生成雾图的参数 (使用平滑后的深度图)
            (imglist[i], img_path, smoothed_depth_maps[i], fog_color, fog_strength, current_hazy_path)
            for i in range(len(imglist)) if smoothed_depth_maps[i] is not None # 只处理有有效平滑深度图的图片
        ]
        with ThreadPoolExecutor(max_workers=num_worker) as executor: # 使用线程池并行执行
            # 执行雾图生成任务并显示进度条
            list(tqdm(executor.map(process_image_for_fog, fog_args), total=len(fog_args), desc=f'{subfolder} - 生成雾图'))

        print(f"雾图生成完成：{current_hazy_path.resolve()}") # 打印完成信息

    if depth_path.exists():
        shutil.rmtree(depth_path) # 生成完成后删除临时深度图目录

    return str(base_output.resolve()) # 返回生成的雾图数据集根目录路径
