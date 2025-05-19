import os
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import traceback
import csv
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, set_start_method

from utils.config import load_config
from utils.create import create_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

set_start_method('spawn', force=True)

# 全局设置 (保持不变)
os.environ['TORCH_HOME'] = './.torch'
RESULT_DIR = 'result/dehaze'
METRICS_DIR = 'result/metrics'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
fog_strengths = [0.5, 0.75]
gt_source_folders = ['M1005', 'M0301', 'M1002', 'M1202', 'M0205', 'M1007']
base_gt_folder = 'data/UAV-M/UAV-benchmark-M'
base_fog_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog'
max_size = 1024
transform_to_tensor = transforms.ToTensor()
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
dehazes = ['NONE', 'DIP', 'FALCON', 'AOD_NET', 'AD_NET', 'FFA']
yaml_path = 'configs/DIP.yaml'

# 图像数据集 (保持不变)
class InferenceImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = transform_to_tensor(img)
        return tensor, path

def process_folder(gpu_id, source_folders, fog_strengths, dehazes, shared_model, return_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = shared_model.to(device)
    model.eval()

    all_metrics_process = []

    for source_folder in source_folders:
        gt_folder_path = os.path.join(base_gt_folder, source_folder)
        for fog_strength in fog_strengths:
            fog_code = f"{int(fog_strength * 100):03d}"
            current_base_fog_folder = f'{base_fog_folder}_{fog_code}'
            current_image_folder_path = os.path.join(current_base_fog_folder, source_folder)
            image_files_for_current_source = sorted([
                f for f in os.listdir(current_image_folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            image_paths = [os.path.join(current_image_folder_path, f) for f in image_files_for_current_source]

            for dehaze in dehazes:
                try:
                    dataset = InferenceImageDataset(image_paths)
                    loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)

                    frame_times_sf = []
                    psnr_list_sf = []
                    ssim_list_sf = []

                    with torch.no_grad():
                        for batch_tensor, batch_paths in tqdm(loader,
                                                                desc=f"[GPU{gpu_id}] {source_folder}-{dehaze}-fog{fog_strength}",
                                                                position=gpu_id, leave=True):
                            batch_tensor = batch_tensor.to(device, non_blocking=True)
                            start_time = time.time()

                            if dehaze == 'NONE':
                                processed_batch = batch_tensor.cpu()
                            else:
                                normalized_input_tensor = F.normalize(batch_tensor, mean=NORM_MEAN.squeeze(0).tolist(),
                                                                      std=NORM_STD.squeeze(0).tolist())
                                output = model.predict(normalized_input_tensor) # 使用 predict 方法
                                processed_batch = torch.clamp(output, 0, 1).cpu() # 假设 predict 返回的是去雾后的归一化图像

                            elapsed = time.time() - start_time
                            frame_times_sf.append(elapsed)

                            for i in range(processed_batch.size(0)):
                                processed_image_for_metrics = processed_batch[i]
                                fog_image_path = batch_paths[i]
                                file_name = os.path.basename(fog_image_path)
                                gt_image_path = os.path.join(gt_folder_path, file_name)
                                try:
                                    gt_image_pil = Image.open(gt_image_path).convert("RGB")
                                    gt_tensor = transform_to_tensor(gt_image_pil)
                                    gt_np = gt_tensor.permute(1, 2, 0).numpy()
                                    processed_np = processed_image_for_metrics.permute(1, 2, 0).numpy()
                                    psnr_val = psnr(gt_np, processed_np, data_range=1)
                                    ssim_val = ssim(gt_np, processed_np, channel_axis=2, data_range=1)
                                    psnr_list_sf.append(psnr_val)
                                    ssim_list_sf.append(ssim_val)
                                except FileNotFoundError:
                                    psnr_list_sf.append(np.nan)
                                    ssim_list_sf.append(np.nan)

                    fps = len(frame_times_sf) / sum(frame_times_sf) if sum(frame_times_sf) > 0 else 0
                    avg_psnr = np.nanmean(psnr_list_sf) if psnr_list_sf else np.nan
                    avg_ssim = np.nanmean(ssim_list_sf) if ssim_list_sf else np.nan

                    all_metrics_process.append({
                        'Fog Strength': fog_strength,
                        'Dehaze Method': dehaze,
                        'Source Folder': source_folder,
                        'FPS': fps,
                        'PSNR': avg_psnr,
                        'SSIM': avg_ssim
                    })

                except Exception as e:
                    print(f"Error processing {source_folder} - {dehaze} - {fog_strength}: {e}")
                    traceback.print_exc()

    return_list.append(all_metrics_process)

def main():
    num_gpus = torch.cuda.device_count()
    available_gpus = list(range(num_gpus))
    num_folders = len(gt_source_folders)
    processes = []
    manager = Manager()
    results = manager.list()

    cfg = load_config(yaml_path)
    model = create_model(cfg)
    model.load_model()
    model.share_memory()

    if not available_gpus:
        print("警告: 没有可用的 CUDA 设备，程序将在 CPU 上串行运行。")
        process_folder('cpu', gt_source_folders, fog_strengths, dehazes, model, results)
    else:
        folders_per_gpu = math.ceil(num_folders / len(available_gpus))
        for i in range(len(available_gpus)):
            start_index = i * folders_per_gpu
            end_index = min((i + 1) * folders_per_gpu, num_folders)
            folders_on_gpu = gt_source_folders[start_index:end_index]
            gpu_id = available_gpus[i]
            process = Process(target=process_folder, args=(gpu_id, folders_on_gpu, fog_strengths, dehazes, model, results))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    all_metrics = []
    for metrics_list in results:
        all_metrics.extend(metrics_list)

    save_metrics(all_metrics, 'dehaze_metrics_parallel.csv')
    print(f"\n所有评估指标已保存到: {os.path.join(METRICS_DIR, 'dehaze_metrics_parallel.csv')}")

def save_metrics(metrics, filename):
    filepath = os.path.join(METRICS_DIR, filename)
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['Fog Strength', 'Dehaze Method', 'Source Folder', 'FPS', 'PSNR', 'SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)

if __name__ == "__main__":
    main()