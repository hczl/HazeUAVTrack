import os
import time
import math
import torch
import numpy as np
import cv2  # 虽然不生成视频，但 cv2 可能在其他地方被依赖，保留
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback  # 导入 traceback 用于打印详细错误

# Assumed to be available
from utils.config import load_config
from utils.create import create_model
# Assumed metric functions are available
from utils.metrics import compute_map, compute_f1, compute_mota
from utils.transform import load_annotations, scale_ground_truth_boxes, scale_ignore_regions

# 设置 Matplotlib 支持中文显示和负号正常显示
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体 (或其他支持中文的字体)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置 PyTorch Hub 的缓存目录
os.environ['TORCH_HOME'] = './.torch'

# Define the mean and std used during training normalization
# These should match the values used in your create_data function
# Use lists/tuples for torchvision.transforms.functional.normalize
NORM_MEAN_LIST = [0.485, 0.456, 0.406]
NORM_STD_LIST = [0.229, 0.224, 0.225]

imgs = '1005'
# ---- 配置参数 ----
# 图像文件夹路径（包含用于评估的图像）
image_folder = f'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog_050/M{imgs}'
# 对应的真实标签文件夹路径
gt_label_folder = f'data/UAV-M/frame_labels/test/M{imgs}'
# 对应的忽略区域文件夹路径
ignore_mask_folder = f'data/UAV-M/frame_ignores/test/M{imgs}'

# 基础模型配置文件路径（用于加载通用配置，如设备、最大尺寸等）
# 这个文件中的 'method']['detector' 字段会在循环中被修改
yaml_path = 'configs/DE_NET.yaml'  # 使用用户提供的原始配置文件路径作为基础

max_size = 1024  # 模型输入图像的最大边长
result_dir = 'result/detector'  # 结果保存目录 (图表、汇总文件等)

# 创建结果目录，如果不存在
os.makedirs(result_dir, exist_ok=True)

transform_to_tensor = transforms.ToTensor()

def preprocess_image(image_pil, max_size, mean, std):
    w, h = image_pil.size
    img_tensor = transform_to_tensor(image_pil)

    r = min(1.0, max_size / float(max(w, h)))
    new_h = max(32, int(math.floor(h * r / 32) * 32))
    new_w = max(32, int(math.floor(w * r / 32) * 32))

    resized_tensor = F.resize(img_tensor, (new_h, new_w))

    normalized_tensor = F.normalize(resized_tensor, mean=mean, std=std)

    input_tensor = normalized_tensor.unsqueeze(0)

    return input_tensor, (w, h), (new_w, new_h)
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


image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
image_paths = [os.path.join(image_folder, f) for f in image_files]

if not image_paths:
    print("错误: 未找到任何图像文件进行评估。")
    exit()

try:
    first_img_pil = Image.open(image_paths[0]).convert('RGB')
    _, orig_size_first, resized_size_first = preprocess_image(first_img_pil, max_size, NORM_MEAN_LIST, NORM_STD_LIST)

    print("加载并缩放真实标签和忽略区域...")
    gt_labels, ignore_masks = load_annotations(gt_label_folder, ignore_mask_folder, len(image_paths))

    gt_labels_scaled = scale_ground_truth_boxes(gt_labels, orig_size_first, resized_size_first)
    ignore_masks_scaled = scale_ignore_regions(ignore_masks, orig_size_first, resized_size_first)

    if len(gt_labels_scaled) != len(image_paths) or len(ignore_masks_scaled) != len(image_paths):
        print(f"警告: 加载的标签/忽略区域数量 ({len(gt_labels_scaled)}/{len(ignore_masks_scaled)}) 与图像数量 ({len(image_paths)}) 不匹配。")

except Exception as e:
    print(f"错误: 加载或处理标签/忽略区域失败: {e}")
    traceback.print_exc()
    gt_labels_scaled = []
    ignore_masks_scaled = []
    print("将无法计算依赖真实标签的指标 (mAP, F1, MOTA等)。")


fps_results = {}
map_results = {}
f1_results = {}
mota_results = {}
motp_results = {}
id_switches_results = {}

# 定义去雾方法列表
dehaze_methods = ["NONE", "FALCON", "AOD_NET", "AD_NET", 'FFA']

# 创建去雾和检测器的组合
detector_dehaze_combinations = []

# 直接添加 DE_NET 和 IA_YOLOV3，不需要去雾
detector_dehaze_combinations.append(('DE_NET', 'NONE'))
detector_dehaze_combinations.append(('IA_YOLOV3', 'NONE'))

# YOLOV11 和 YOLOV3 需要与所有去雾方法组合
for detector in ['YOLOV11', 'YOLOV3']:
    for dehaze in dehaze_methods:
        detector_dehaze_combinations.append((detector, dehaze))

# 对每种组合进行循环评估
for detector_name, dehaze_method in detector_dehaze_combinations:
    print(f"\n加载检测器: {detector_name} 与去雾方法: {dehaze_method}")
    try:
        cfg = load_config(yaml_path)
    except Exception as e:
        print(f"错误: 加载配置文件 {yaml_path} 失败: {e}")
        traceback.print_exc()
        print(f"跳过检测器 {detector_name} 与去雾方法 {dehaze_method}")
        continue

    if 'method' not in cfg or 'detector' not in cfg['method']:
        print(f"警告: 配置文件 {yaml_path} 没有 'method.detector' 键。无法设置检测器名称。")
        print(f"跳过检测器 {detector_name} 与去雾方法 {dehaze_method}")
        continue

    # 设置检测器名称和去雾方法
    cfg['method']['detector'] = detector_name
    cfg['method']['dehaze'] = dehaze_method

    try:
        model = create_model(cfg)
        device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        try:
            model.load_model()
        except Exception as e:
            print(f"警告: 加载模型权重失败 for {detector_name}: {e}")
            print("将使用未加载权重的模型。请检查 load_model 方法或权重路径。")
            traceback.print_exc()

    except Exception as e:
        print(f"错误: 创建或初始化模型 {detector_name} 与去雾方法 {dehaze_method} 失败: {e}")
        traceback.print_exc()
        print(f"跳过检测器 {detector_name} 与去雾方法 {dehaze_method}")
        continue

    frame_times = []
    current_preds = []

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths, desc=f"评估 {detector_name} 与去雾方法 {dehaze_method}")):
            try:
                img_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"\n警告: 跳过文件: {img_path} ({e})")
                traceback.print_exc()
                current_preds.append([])
                continue

            input_tensor, _, _ = preprocess_image(img_pil, max_size, NORM_MEAN_LIST, NORM_STD_LIST)
            input_tensor = input_tensor.to(device)

            try:
                processed_img_tensor = input_tensor

                start_time = time.time()
                preds = model.predict(processed_img_tensor)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                frame_times.append(elapsed)

                if isinstance(preds, np.ndarray):
                    preds = preds.tolist()

                conf_threshold = 0.82
                filtered_boxes = [box for box in preds if len(box) > 5 and box[5] >= conf_threshold]

                current_preds.append(filtered_boxes)

            except Exception as e:
                print(f"\n错误: 处理图像 {image_files[idx]} 失败: {e}")
                traceback.print_exc()
                current_preds.append([])

    print(f"\n计算 {detector_name} 与去雾方法 {dehaze_method} 的性能指标...")

    valid_frame_times = [t for t in frame_times if t is not None]
    if valid_frame_times:
        avg_fps = len(valid_frame_times) / sum(valid_frame_times)
    else:
        avg_fps = 0
        print(f"警告: {detector_name} 与去雾方法 {dehaze_method} 没有成功处理任何帧进行计时。")

    min_len = min(len(current_preds), len(gt_labels_scaled), len(ignore_masks_scaled))
    if min_len < len(image_paths):
        print(f"警告: 预测、GT或忽略区域列表长度不一致 ({len(current_preds)}, {len(gt_labels_scaled)}, {len(ignore_masks_scaled)}). 仅使用前 {min_len} 帧进行评估。")

    current_preds_eval = current_preds[:min_len]
    gt_labels_scaled_eval = gt_labels_scaled[:min_len]
    ignore_masks_scaled_eval = ignore_masks_scaled[:min_len]

    if not current_preds_eval or not gt_labels_scaled_eval:
        print(f"警告: 没有有效的预测或真实标签，无法计算依赖GT的指标 for {detector_name} 与去雾方法 {dehaze_method}.")
        avg_map = 0
        avg_f1 = 0
        avg_mota = 0
        avg_motp = 0
        total_id_switches = 0
    else:
        try:
            avg_map = compute_map(current_preds_eval, gt_labels_scaled_eval, ignore_masks=ignore_masks_scaled_eval)
            avg_f1 = compute_f1(current_preds_eval, gt_labels_scaled_eval, ignore_masks=ignore_masks_scaled_eval)
            avg_mota, avg_motp, total_id_switches = compute_mota(current_preds_eval, gt_labels_scaled_eval, ignore_masks=ignore_masks_scaled_eval)
        except Exception as e:
            print(f"错误: 计算指标失败 for {detector_name} 与去雾方法 {dehaze_method}: {e}")
            traceback.print_exc()
            avg_map = 0
            avg_f1 = 0
            avg_mota = 0
            avg_motp = 0
            total_id_switches = 0
            print("将指标设置为 0。")

    # 如果去雾方法是 NONE，使用简单的命名
    result_key = f"{detector_name}" if dehaze_method == "NONE" else f"{dehaze_method}_{detector_name}"

    fps_results[result_key] = avg_fps
    map_results[result_key] = avg_map
    f1_results[result_key] = avg_f1
    mota_results[result_key] = avg_mota
    motp_results[result_key] = avg_motp
    id_switches_results[result_key] = total_id_switches

    print(f"\n--- {detector_name} 与去雾方法 {dehaze_method} 评估结果 ---")
    print(f"平均 FPS (仅预测): {avg_fps:.2f}")
    print(f"平均精度 (mAP): {avg_map:.4f}")
    print(f"F1 得分: {avg_f1:.4f}")
    print(f"MOTA: {avg_mota:.4f}")
    print(f"MOTP: {avg_motp:.4f}")
    print(f"ID切换次数: {total_id_switches}")
    print("-" * (len(detector_name) + 10))

# 生成评估结果图表
print("\n生成评估结果图表...")
plot_metric(fps_results, "各检测器的平均 FPS", "FPS", "detector_fps.png")
plot_metric(map_results, "各检测器的平均 mAP", "mAP", "detector_map.png")
plot_metric(f1_results, "各检测器的平均 F1 得分", "F1", "detector_f1.png")
plot_metric(mota_results, "各检测器的 MOTA", "MOTA", "detector_mota.png", reverse_sort=True)
plot_metric(motp_results, "各检测器的 MOTP", "MOTP", "detector_motp.png", reverse_sort=False)
plot_metric(id_switches_results, "各检测器的总 ID 切换次数", "ID 切换次数", "detector_id_switches.png")

# 写入汇总文件
summary_path = os.path.join(result_dir, "detector_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("--- Detector Performance Evaluation Summary ---\n")
    f.write(f"Source Image Directory: {image_folder}\n")
    f.write(f"Ground Truth Label Directory: {gt_label_folder}\n")
    f.write(f"Ignore Regions Directory: {ignore_mask_folder}\n")
    f.write(f"Evaluated Detectors: {', '.join([f'{detector}_{dehaze}' for detector, dehaze in detector_dehaze_combinations])}\n")
    f.write(f"Normalization Parameters: Mean={NORM_MEAN_LIST}, Std={NORM_STD_LIST}\n")
    f.write("-" * 30 + "\n\n")

    f.write("平均 FPS (仅预测时间):\n")
    for name, value in sorted(fps_results.items(), key=lambda item: item[1], reverse=True):
        f.write(f"  {name}: {value:.2f}\n")
    f.write("\n")

    f.write("平均精度 (mAP):\n")
    for name, value in sorted(map_results.items(), key=lambda item: item[1], reverse=True):
        f.write(f"  {name}: {value:.4f}\n")
    f.write("\n")

    f.write("F1 得分:\n")
    for name, value in sorted(f1_results.items(), key=lambda item: item[1], reverse=True):
        f.write(f"  {name}: {value:.4f}\n")
    f.write("\n")

    f.write("MOTA:\n")
    for name, value in sorted(mota_results.items(), key=lambda item: item[1], reverse=True):
        f.write(f"  {name}: {value:.4f}\n")
    f.write("\n")

    f.write("MOTP:\n")
    for name, value in sorted(motp_results.items(), key=lambda item: item[1], reverse=False):
        f.write(f"  {name}: {value:.4f}\n")
    f.write("\n")

    f.write("总 ID 切换次数:\n")
    for name, value in sorted(id_switches_results.items(), key=lambda item: item[1], reverse=False):
        f.write(f"  {name}: {int(value)}\n")
    f.write("\n")

print(f"汇总结果已保存到: {summary_path}")
print("\n所有检测器评估完成。")