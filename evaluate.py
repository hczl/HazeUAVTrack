import os
import time
import math
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from utils.config import load_config
from utils.create import create_model
from utils.metrics import compute_map, compute_f1, compute_mota
from utils.transform import load_annotations, scale_ground_truth_boxes, scale_ignore_regions


def preprocess(image_pil):
    w, h = image_pil.size
    img_tensor = transform(image_pil)
    r = min(1.0, max_size / float(max(w, h)))
    new_h = max(32, int(math.floor(h * r / 32) * 32))
    new_w = max(32, int(math.floor(w * r / 32) * 32))
    resized = F.resize(img_tensor, (new_h, new_w))
    # print((w, h), (new_w, new_h))
    return resized.unsqueeze(0).to(device), (w, h), (new_w, new_h)

# ---- 配置参数 ----
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1005'
yaml_path = 'configs/DE_NET.yaml'
output_fps = 30
max_size = 640
# 加载真实标签和忽略区域

# ---- 加载模型和配置 ----
cfg = load_config(yaml_path)
model = create_model(cfg)
model.load_model()
device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
model.to(device).eval()

model_name = cfg['method'].get('detector', 'unnamed_detector')
output_dir = os.path.join('result', 'detector', model_name)
os.makedirs(output_dir, exist_ok=True)
video_path = os.path.join(output_dir, 'detection_video.mp4')
result_txt = os.path.join(output_dir, 'detections.txt')

print(f"检测结果将保存至: {output_dir}")

# ---- 图像预处理 ----
transform = transforms.ToTensor()




# ---- 图像列表 ----
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
gt_labels, ignore_masks = load_annotations(
    'data/UAV-M/frame_labels/test/M1005',
    'data/UAV-M/frame_ignores/test/M1005',
    len(image_files)
)
# 用第一帧初始化时保存尺寸
first_img = Image.open(image_files[0])
_, orig_size, resized_size = preprocess(first_img)

# ---- 视频写入准备 ----
out = None
all_preds = []
all_gts = []  # 若有标注数据，可加载GT用于评估

gt_labels = scale_ground_truth_boxes(gt_labels, orig_size, resized_size)
ignore_masks = scale_ignore_regions(ignore_masks, orig_size, resized_size)
all_time = 0

with open(result_txt, 'w') as f_out, torch.no_grad():
    for idx, img_path in enumerate(tqdm(image_files, desc="检测中")):
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            continue

        input_tensor, orig_size, resized_size = preprocess(img_pil)
        dehazed = model.dehaze(input_tensor)
        frame_np = (dehazed[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        s_t = time.time()
        preds = model.predict(input_tensor)
        e_t = time.time()
        all_time += e_t - s_t
        # boxes = scale_boxes(preds, orig_size, resized_size)
        boxes = [box for box in preds if box[4] >= cfg['conf_threshold']]

        all_preds.append(boxes)

        # ---- 画检测框 ----
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame_np, f"{conf:.2f}", (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            f_out.write(f"{idx},{x1},{y1},{x2},{y2},{conf:.4f}\n")

        # ---- 新增：画忽略区域框（紫色）----
        if idx < len(ignore_masks):
            for ig_box in ignore_masks[idx]:
                x1, y1, x2, y2 = map(int, ig_box)
                cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 紫色 BGR: (255, 0, 255)

        # ---- 视频写入 ----
        if out is None:
            h, w, _ = frame_np.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, output_fps, (w, h))
            if not out.isOpened():
                print("无法初始化视频写入对象。")
                out = None

        if out:
            out.write(frame_np)
# with open(result_txt, 'w') as f_out:
#     for idx, img_path in enumerate(tqdm(image_files, desc="使用标签生成视频")):
#         try:
#             img_pil = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             print(f"无法读取图像 {img_path}: {e}")
#             continue
#
#         input_tensor, orig_size, resized_size = preprocess(img_pil)
#         frame_np = (input_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#         frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
#
#         # 使用 ground truth 框作为“预测”
#         boxes = gt_labels[idx]
#         all_preds.append(boxes)
#
#         # ---- 画GT框（作为预测使用）----
#         for box in boxes:
#             x1, y1, x2, y2, obj_id = map(int, box)
#             conf = 1.0  # 假定置信度为 1.0
#             cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame_np, f"GT-{obj_id}", (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#             f_out.write(f"{idx},{x1},{y1},{x2},{y2},{conf:.4f},{obj_id}\n")
#
#         # ---- 画忽略区域框（紫色）----
#         if idx < len(ignore_masks):
#             for ig_box in ignore_masks[idx]:
#                 x1, y1, x2, y2 = map(int, ig_box)
#                 cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 255), 2)
#
#         # ---- 视频写入 ----
#         if out is None:
#             h, w, _ = frame_np.shape
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(video_path, fourcc, output_fps, (w, h))
#             if not out.isOpened():
#                 print("无法初始化视频写入对象。")
#                 out = None
#
#         if out:
#             out.write(frame_np)

for frame_i, (preds, gts, ignores) in enumerate(zip(all_preds, gt_labels, ignore_masks)):
    print(f"Frame {frame_i}: {len(preds)} preds, {len(gts)} gts, {len(ignores)} ignores")


if out:
    out.release()

# ---- 性能评估 ----
end_time = time.time()
fps = len(image_files) / all_time
print(f"\n系统处理速度: {fps:.2f} FPS")



# 评估指标（示例）
map_score = compute_map(all_preds, gt_labels, ignore_masks=ignore_masks)
f1_score = compute_f1(all_preds, gt_labels, ignore_masks=ignore_masks)
mota_score, motp_score, id_switches = compute_mota(all_preds, gt_labels, ignore_masks=ignore_masks)

print(f"平均精度 (mAP): {map_score:.4f}")
print(f"F1 得分: {f1_score:.4f}")
print(f"MOTA: {mota_score:.4f}")
print(f"MOTP: {motp_score:.4f}")
print(f"ID切换次数: {id_switches}")


print(f"检测框保存至: {result_txt}")
print(f"视频已保存至: {video_path}")
print("处理完成。")
