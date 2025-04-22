import torch
import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def haze(input_path):
    input_path = Path(input_path)
    print(input_path)
    dataset_name = input_path.name
    base_output = Path(f"MiDaS_Deep_{dataset_name}")
    depth_path = base_output / 'depth_temp'
    # 清空旧深度图
    if depth_path.exists():
        shutil.rmtree(depth_path)
    hazy_path = base_output
    if os.path.exists(input_path.parent / f"MiDaS_Deep_{dataset_name}"):
        return input_path.parent / f"MiDaS_Deep_{dataset_name}"


    # 模型加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = 'DPT_Large'
    torch.hub.set_dir('/path/to/custom/cache')  # 可自定义
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas.to(device)
    midas.eval()

    # 雾气参数
    fog_strength = 1.0
    fog_color = np.array([200, 200, 200], dtype=np.uint8)

    subfolders = [f for f in os.listdir(input_path) if (input_path / f).is_dir()]

    for subfolder in subfolders:
        img_path = input_path / subfolder
        current_depth_path = depth_path / subfolder
        current_hazy_path = hazy_path / subfolder
        os.makedirs(current_depth_path, exist_ok=True)
        os.makedirs(current_hazy_path, exist_ok=True)

        imglist = sorted(os.listdir(img_path))
        with tqdm(total=len(imglist), desc=f'{subfolder} - 深度图') as pbar:
            for img_name in imglist:
                img_file = img_path / img_name
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform
                input_tensor = transform(image).to(device)

                with torch.no_grad():
                    prediction = midas(input_tensor)
                depth_map = prediction.squeeze().cpu().numpy()
                depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                resized_depth = cv2.resize(depth_map_norm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(current_depth_path / img_name), resized_depth)

                pbar.update(1)

        with tqdm(total=len(imglist), desc=f'{subfolder} - 生成雾图') as pbar:
            for img_name in imglist:
                img_file = img_path / img_name
                depth_file = current_depth_path / img_name

                image = cv2.imread(str(img_file))
                depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
                if image is None or depth is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                depth_norm = depth.astype(np.float32) / 255
                fog_intensity = np.clip(1 - depth_norm, 0, 1) * fog_strength

                fog_layer = np.ones_like(image, dtype=np.float32) * fog_color
                foggy = image * (1 - fog_intensity[:, :, None]) + fog_layer * fog_intensity[:, :, None]
                foggy = np.clip(foggy, 0, 255).astype(np.uint8)

                output_file = current_hazy_path / img_name
                cv2.imwrite(str(output_file), cv2.cvtColor(foggy, cv2.COLOR_RGB2BGR))

                pbar.update(1)

    # 删除临时深度图
    if depth_path.exists():
        shutil.rmtree(depth_path)

    return str(hazy_path.resolve())
