import torch
import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_image_for_depth(args):
    img_name, img_path, transform, midas, device, current_depth_path = args
    img_file = img_path / img_name
    image = cv2.imread(str(img_file))
    if image is None:
        return img_name, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized_depth = cv2.resize(depth_map_norm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(str(current_depth_path / img_name), resized_depth)
    return img_name, resized_depth

def process_image_for_fog(args):
    img_name, img_path, depth, fog_color, fog_strength, current_hazy_path = args
    img_file = img_path / img_name
    image = cv2.imread(str(img_file))
    if image is None or depth is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_norm = depth.astype(np.float32) / 255
    fog_intensity = np.clip(1 - depth_norm, 0, 1) * fog_strength
    fog_layer = np.ones_like(image, dtype=np.float32) * fog_color
    foggy = image * (1 - fog_intensity[:, :, None]) + fog_layer * fog_intensity[:, :, None]
    foggy = np.clip(foggy, 0, 255).astype(np.uint8)
    output_file = current_hazy_path / img_name
    cv2.imwrite(str(output_file), cv2.cvtColor(foggy, cv2.COLOR_RGB2BGR))

def MiDaS_Deep(input_path, fog_strength, num_worker=8):
    input_path = Path(input_path)
    dataset_name = input_path.name
    fog_strength_str = f"fog_{int(fog_strength * 100):03d}"
    base_output = input_path.parent / Path(f"MiDaS_Deep_{dataset_name}_{fog_strength_str}")
    depth_path = base_output / 'depth_temp'
    if depth_path.exists():
        shutil.rmtree(depth_path)
    if os.path.exists(base_output):
        return str(base_output.resolve())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = 'DPT_Large'
    torch.hub.set_dir('/path/to/custom/cache')  # 可自定义缓存
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas.to(device)
    midas.eval()

    fog_color = np.array([200, 200, 200], dtype=np.uint8)
    subfolders = [f for f in os.listdir(input_path) if (input_path / f).is_dir()]

    for subfolder in subfolders:
        img_path = input_path / subfolder
        current_depth_path = depth_path / subfolder
        current_hazy_path = base_output / subfolder
        os.makedirs(current_depth_path, exist_ok=True)
        os.makedirs(current_hazy_path, exist_ok=True)

        imglist = sorted(os.listdir(img_path))
        transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

        # 并行计算深度图
        depth_maps = [None] * len(imglist)
        args = [(img_name, img_path, transform, midas, device, current_depth_path) for img_name in imglist]
        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            for result in tqdm(executor.map(process_image_for_depth, args), total=len(args), desc=f'{subfolder} - 深度图'):
                img_name, depth = result
                idx = imglist.index(img_name)
                depth_maps[idx] = depth

        # 滑动平均平滑深度图
        smoothed_depth_maps = []
        window_size = 5
        for i in range(len(depth_maps)):
            if depth_maps[i] is None:
                smoothed_depth_maps.append(None)
                continue
            start = max(0, i - window_size // 2)
            end = min(len(depth_maps), i + window_size // 2 + 1)
            window = [d for d in depth_maps[start:end] if d is not None]
            smoothed = np.mean(window, axis=0)
            smoothed_depth_maps.append(smoothed.astype(np.uint8))

        # 并行生成雾图
        fog_args = [
            (imglist[i], img_path, smoothed_depth_maps[i], fog_color, fog_strength, current_hazy_path)
            for i in range(len(imglist))
        ]
        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            list(tqdm(executor.map(process_image_for_fog, fog_args), total=len(fog_args), desc=f'{subfolder} - 生成雾图'))

        print(f"雾图生成完成：{current_hazy_path.resolve()}")

    if depth_path.exists():
        shutil.rmtree(depth_path)

    return str(base_output.resolve())
