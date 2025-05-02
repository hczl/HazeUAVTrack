import os
import time
import cv2
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from models.trackers.deep_sort_pytorch.deep_sort import DeepSort
from utils.common import create_model
from utils.config import load_config

# ---- 配置 ----
use_deepsort = True  # 设置为 False 时只使用检测框，不使用 DeepSORT 跟踪
os.environ['TORCH_HOME'] = './.torch'
cfg = load_config('configs/exp3.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# ---- 初始化模型 ----
model = create_model(cfg)
model.load_checkpoint(f"models/{cfg['detector']}/checkpoints/best_model.pth")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")

# ---- 初始化 DeepSORT ----
if use_deepsort:
    deepsort = DeepSort(model_path='models/trackers/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')

# ---- 视频参数 ----
output_path = f"experiments/{cfg['experiment_name']}/results/output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None
frame_times = []

# ---- 图像读取与预处理 ----
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M0101'
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
transform = transforms.Compose([transforms.ToTensor()])
max_size = 608
temp_video_frames = []

# ---- 推理与处理循环 ----
with torch.no_grad():
    for idx, image_path in enumerate(image_files):
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"跳过无法打开的图片: {image_path}")
            continue

        start_time = time.time()

        image_tensor = transform(image)
        orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
        r = min(1.0, max_size / float(max(orig_w, orig_h)))
        new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
        new_w = max(32, int(math.floor(orig_w * r / 32) * 32))
        image_resized = F.resize(image_tensor, (new_h, new_w))
        input_tensor = image_resized.unsqueeze(0).to(device)

        detections = model.predict(input_tensor, ignore=None, conf_thresh=0.5)

        image_np = image_resized.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        result_image = image_np.copy()

        if len(detections) > 0 and len(detections[0]) > 0:
            dets = np.array(detections[0])
            bboxes = dets[:, :4]
            scores = dets[:, 4]

            if use_deepsort:
                dummy_classes = np.ones(len(bboxes), dtype=int)
                outputs, _ = deepsort.update(bboxes, scores, dummy_classes, result_image)

                if len(outputs) > 0:
                    for out in outputs:
                        if len(out) == 6:
                            x1, y1, x2, y2, cls_id, track_id = map(int, out)
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(result_image, f"ID: {track_id}", (x1, y1 - 7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(result_image, "det", (x1, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        elapsed = time.time() - start_time
        frame_times.append(elapsed)

        if video_writer is None:
            h, w = result_image.shape[:2]

        temp_video_frames.append(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

# ---- 视频写入阶段 ----
if frame_times:
    avg_fps = len(frame_times) / sum(frame_times)
    print(f"\n平均 FPS：{avg_fps:.2f}")

    video_writer = cv2.VideoWriter(output_path, fourcc, avg_fps, (w, h))
    for frame in temp_video_frames:
        video_writer.write(frame)
    video_writer.release()
    print(f"视频已保存至: {output_path}")
else:
    print("未生成视频，可能图片路径异常。")
