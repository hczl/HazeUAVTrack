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

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

os.environ['TORCH_HOME'] = './.torch'
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1005'
gt_folder = 'data/UAV-M/UAV-benchmark-M/M1005'
max_size = 640
result_dir = 'result/dehaze'
video_dir = 'result/video'

os.makedirs(result_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

transform = transforms.ToTensor()
dehazes = ['DIP', 'DENET', 'FALCON','AOD_NET', 'BDN']
yaml_path = 'configs/DIP.yaml'

fps_results = {}
psnr_results = {}
ssim_results = {}

image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for dehaze in dehazes:
    print(f"\n加载去雾模型: {dehaze}")
    cfg = load_config(yaml_path)
    cfg['method']['dehaze'] = dehaze
    model = create_model(cfg)
    model.load_model()
    device = cfg['device']
    model.to(device)

    frame_times = []
    psnr_list = []
    ssim_list = []
    processed_frames = []

    with torch.no_grad():
        for file_name in image_files:
            image_path = os.path.join(image_folder, file_name)
            gt_path = os.path.join(gt_folder, file_name)

            try:
                image = Image.open(image_path).convert("RGB")
                gt_image = Image.open(gt_path).convert("RGB")
            except:
                print(f"跳过文件: {file_name}")
                continue

            image_tensor = transform(image)
            gt_tensor = transform(gt_image)

            orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
            r = min(1.0, max_size / float(max(orig_w, orig_h)))
            new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
            new_w = max(32, int(math.floor(orig_w * r / 32) * 32))
            image_resized = F.resize(image_tensor, (new_h, new_w))
            gt_resized = F.resize(gt_tensor, (new_h, new_w))

            input_tensor = image_resized.unsqueeze(0).to(device)

            start_time = time.time()
            output = model.predict(input_tensor)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            frame_times.append(elapsed)

            output_image = output.squeeze(0).cpu().clamp(0, 1)

            psnr_val = psnr(gt_resized.permute(1, 2, 0).numpy(), output_image.permute(1, 2, 0).numpy(), data_range=1)
            ssim_val = ssim(gt_resized.permute(1, 2, 0).numpy(), output_image.permute(1, 2, 0).numpy(), channel_axis=2, data_range=1)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            processed_np = (output_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            processed_frames.append(processed_np)

    if frame_times:
        fps = len(frame_times) / sum(frame_times)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        fps_results[dehaze] = fps
        psnr_results[dehaze] = avg_psnr
        ssim_results[dehaze] = avg_ssim

        print(f"{dehaze} - FPS: {fps:.2f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")

        if processed_frames:
            print(f"正在生成视频: {dehaze}.mp4")
            video_path = os.path.join(video_dir, f'{dehaze}.mp4')
            height, width, _ = processed_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            for frame in processed_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"视频已保存到: {video_path}")

    else:
        fps_results[dehaze] = 0
        psnr_results[dehaze] = 0
        ssim_results[dehaze] = 0
        print(f"{dehaze} 未处理图像。")

def plot_metric(result_dict, title, ylabel, filename):
    items = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*items)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(values) * 1.2)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

plot_metric(fps_results, "各去雾方法的平均 FPS", "FPS", "fps.png")
plot_metric(psnr_results, "各去雾方法的平均 PSNR", "PSNR", "psnr.png")
plot_metric(ssim_results, "各去雾方法的平均 SSIM", "SSIM", "ssim.png")
