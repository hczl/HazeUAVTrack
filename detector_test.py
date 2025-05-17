import os
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback
import pandas as pd
from utils.config import load_config
from utils.create import create_model
from utils.metrics import compute_map, compute_f1, compute_mota
from utils.transform import load_annotations, scale_ground_truth_boxes, scale_ignore_regions

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
os.environ['TORCH_HOME'] = './.torch'

TRACKER_NAMES = ['strongsort', 'deepocsort', 'botsort']
TRACKER_THRESHOLDS = {
    'StrongSORT': (0.6, 0.5),
    'DeepOCSORT': (0.65, 0.55),
    'BoT-SORT': (0.7, 0.6)
}

NORM_MEAN_LIST = [0.485, 0.456, 0.406]
NORM_STD_LIST = [0.229, 0.224, 0.225]
source_folders = ['M1005', 'M0301', 'M1002', 'M1202', 'M0205', 'M1007']
base_gt_label_folder = 'data/UAV-M/frame_labels/test'
base_ignore_mask_folder = 'data/UAV-M/frame_ignores/test'
base_fog_image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog'
fog_strengths = [0.5, 0.75]
yaml_path = 'configs/DE_NET.yaml'
max_size = 1024
result_dir = 'result/detector_combined'
os.makedirs(result_dir, exist_ok=True)
transform_to_tensor = transforms.ToTensor()


def preprocess_image(image_pil, max_size, mean, std):
    w, h = image_pil.size
    img_tensor = transform_to_tensor(image_pil)
    r = min(1.0, max_size / float(max(w, h)))
    new_h = max(32, int(math.floor(h * r / 32) * 32))
    new_w = max(32, int(math.floor(w * r / 32) * 32))
    resized_tensor = F.resize(img_tensor, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
    normalized_tensor = F.normalize(resized_tensor, mean=mean, std=std)
    input_tensor = normalized_tensor.unsqueeze(0)
    return input_tensor, (w, h), (new_w, new_h)


def plot_metric(result_dict, title, ylabel, filename, save_dir, is_integer=False, reverse_sort=True):
    if not result_dict:
        print(f"没有数据可绘制 {title}。")
        return
    sortable_items = []
    for name, value in result_dict.items():
        sort_value = value if not (isinstance(value, float) and math.isnan(value)) else (
            -float('inf') if reverse_sort else float('inf'))
        sortable_items.append((sort_value, name, value))
    sorted_items = sorted(sortable_items, key=lambda item: (item[0], item[1]), reverse=reverse_sort)
    names = [item[1] for item in sorted_items]
    values = [item[2] for item in sorted_items]
    valid_values_for_plot = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not valid_values_for_plot:
        print(f"没有有效数据可绘制 {title}。")
        return
    plt.figure(figsize=(12, 8))
    bars = plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    min_val = min(valid_values_for_plot)
    max_val = max(valid_values_for_plot)
    padding = (max_val - min_val) * 0.1 if max_val != min_val else abs(max_val) * 0.1 if max_val != 0 else 0.1
    if 'MOTA' in title:
        lower_bound = min(0.0, min_val - padding)
        upper_bound = max(0.1, max_val + padding)
    elif 'ID 切换次数' in title:
        lower_bound = max(0.0, min_val - padding)
        upper_bound = max(1.0, max_val + padding)
        if any(v > 0 for v in valid_values_for_plot):
            upper_bound = max(upper_bound, max_val * 1.2)
    elif 'mAP' in title or 'F1' in title or 'MOTP' in title:
        lower_bound = max(0.0, min_val - padding)
        upper_bound = max(0.1, max_val + padding)
    else:
        lower_bound = max(0.0, min_val - padding)
        upper_bound = max(0.1, max_val + padding)
    plt.ylim(lower_bound, upper_bound)
    y_range = plt.ylim()[1] - plt.ylim()[0]
    vertical_offset = y_range * 0.02 if y_range > 0 else 0.02
    for bar in bars:
        h = bar.get_height()
        if math.isnan(h):
            label_text = "NaN"
        elif is_integer:
            label_text = f"{int(h)}"
        else:
            label_text = f"{h:.2f}" if abs(h) >= 0.01 or h == 0 else f"{h:.4f}"
        if not math.isnan(h):
            text_y = h + (vertical_offset if h >= 0 else -vertical_offset)
            text_y = max(plt.ylim()[0], min(plt.ylim()[1], text_y))
            plt.text(bar.get_x() + bar.get_width() / 2, text_y,
                     label_text, ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, plt.ylim()[0] + vertical_offset, label_text, ha='center',
                     va='bottom', color='gray', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存到: {save_path}")


detector_dehaze_combinations = [("AD_YOLOV11", "NONE"), ("DE_NET", "NONE"), ("IA_YOLOV3", "NONE"), ("YOLOV3", "NONE"),
                                ("YOLOV11", "NONE")]

for current_tracker_name in TRACKER_NAMES:
    print(f"\n################ 开始处理追踪器: {current_tracker_name} ################")
    tracker_base_result_dir = os.path.join(result_dir, current_tracker_name)
    os.makedirs(tracker_base_result_dir, exist_ok=True)

    combined_results_for_current_tracker = {}

    for fog_strength in fog_strengths:
        fog_code = f"{int(fog_strength * 100):03d}"
        current_base_fog_image_folder = f'{base_fog_image_folder}_{fog_code}'
        print(f"\n======== 正在处理雾浓度: {fog_strength} (追踪器: {current_tracker_name}) ========")
        print(f"使用浓雾图像文件夹: {current_base_fog_image_folder}")

        tracker_fog_result_dir = os.path.join(tracker_base_result_dir, f'fog_{fog_strength}')
        os.makedirs(tracker_fog_result_dir, exist_ok=True)

        threshold_index = 0 if fog_strength == 0.5 else 1
        active_conf_threshold = TRACKER_THRESHOLDS[current_tracker_name][threshold_index]
        print(f"追踪器 {current_tracker_name} 在雾浓度 {fog_strength} 下使用置信度阈值: {active_conf_threshold}")

        for detector_name, dehaze_method in detector_dehaze_combinations:
            result_key = f"{detector_name}" if dehaze_method == "NONE" else f"{dehaze_method}_{detector_name}"
            print(
                f"\n-------- 正在处理组合: {result_key} (追踪器: {current_tracker_name}, 雾浓度: {fog_strength}) --------")

            source_fps_list = []
            source_map_list = []
            source_f1_list = []
            source_mota_list = []
            source_motp_list = []
            source_id_switches_list = []

            model = None
            cfg = None
            device = "cpu"
            combination_load_failed = False

            try:
                cfg = load_config(yaml_path)
                if 'method' not in cfg:
                    cfg['method'] = {}
                cfg['method']['detector'] = detector_name
                cfg['method']['dehaze'] = dehaze_method
                cfg['method']['track_method'] = current_tracker_name

                model = create_model(cfg)
                device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
                model.load_model()
                model.to(device)
                model.eval()

            except Exception as e:
                print(
                    f"错误: 为组合 {result_key} (追踪器: {current_tracker_name}, 雾浓度: {fog_strength}) 创建或初始化模型失败: {e}")
                traceback.print_exc()
                combination_load_failed = True
                print("跳过此组合在此雾浓度下的评估。")

            for source_folder in source_folders:
                print(
                    f"\n---- 正在处理源文件夹: {source_folder} (组合: {result_key}, 追踪器: {current_tracker_name}, 雾浓度: {fog_strength}) ----")
                image_folder_path = os.path.join(current_base_fog_image_folder, source_folder)
                gt_label_folder_path = os.path.join(base_gt_label_folder, source_folder)
                ignore_mask_folder_path = os.path.join(base_ignore_mask_folder, source_folder)

                if not os.path.exists(image_folder_path):
                    print(
                        f"警告: 浓雾图像文件夹未找到: {image_folder_path}。跳过源 {source_folder} (组合: {result_key})。")
                    source_fps_list.append(np.nan)
                    source_map_list.append(np.nan)
                    source_f1_list.append(np.nan)
                    source_mota_list.append(np.nan)
                    source_motp_list.append(np.nan)
                    source_id_switches_list.append(np.nan)
                    continue

                if not os.path.exists(gt_label_folder_path):
                    print(f"警告: GT标注文件夹未找到: {gt_label_folder_path}。无法计算源 {source_folder} 的GT相关指标。")
                    gt_label_folder_path = None
                if not os.path.exists(ignore_mask_folder_path):
                    print(f"警告: 忽略区域掩码文件夹未找到: {ignore_mask_folder_path}。")
                    ignore_mask_folder_path = None

                image_files = sorted(
                    [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                image_paths_sf = [os.path.join(image_folder_path, f) for f in image_files]

                if not image_paths_sf:
                    print(f"警告: 在 {image_folder_path} 中未找到图像。跳过源 {source_folder} (组合: {result_key})。")
                    source_fps_list.append(np.nan)
                    source_map_list.append(np.nan)
                    source_f1_list.append(np.nan)
                    source_mota_list.append(np.nan)
                    source_motp_list.append(np.nan)
                    source_id_switches_list.append(np.nan)
                    continue

                gt_labels_scaled_sf = []
                ignore_masks_scaled_sf = []
                can_compute_gt_metrics = False

                if gt_label_folder_path and ignore_mask_folder_path:
                    try:
                        first_img_pil_sf = Image.open(image_paths_sf[0]).convert('RGB')
                        _, orig_size_sf, resized_size_sf = preprocess_image(first_img_pil_sf, max_size, NORM_MEAN_LIST,
                                                                            NORM_STD_LIST)
                        gt_labels_sf, ignore_masks_sf = load_annotations(gt_label_folder_path, ignore_mask_folder_path,
                                                                         len(image_paths_sf))
                        gt_labels_scaled_sf = scale_ground_truth_boxes(gt_labels_sf, orig_size_sf, resized_size_sf)
                        ignore_masks_scaled_sf = scale_ignore_regions(ignore_masks_sf, orig_size_sf, resized_size_sf)
                        if len(gt_labels_scaled_sf) == len(image_paths_sf) and len(ignore_masks_scaled_sf) == len(
                                image_paths_sf):
                            can_compute_gt_metrics = True
                            print(f"成功加载并缩放源 {source_folder} 的GT/忽略区域。")
                        else:
                            print(
                                f"警告: 源 {source_folder} 的GT/忽略区域数量不匹配 ({len(gt_labels_scaled_sf)}/{len(ignore_masks_scaled_sf)} vs {len(image_paths_sf)} 图像)。无法计算此源的GT相关指标。")
                            gt_labels_scaled_sf = []
                            ignore_masks_scaled_sf = []
                    except Exception as e:
                        print(f"错误: 加载/缩放源 {source_folder} 的GT/忽略区域失败: {e}")
                        traceback.print_exc()
                        gt_labels_scaled_sf = []
                        ignore_masks_scaled_sf = []
                        print("无法计算此源文件夹的GT相关指标。")
                elif gt_label_folder_path is None or ignore_mask_folder_path is None:
                    print(f"由于缺少GT/忽略文件夹，跳过源 {source_folder} 的GT相关指标计算。")

                frame_times_sf = []
                current_preds_sf = []

                if combination_load_failed:
                    print(f"由于模型加载失败，跳过源 {source_folder} 的图像处理。")
                    source_fps_list.append(np.nan)
                    source_map_list.append(np.nan)
                    source_f1_list.append(np.nan)
                    source_mota_list.append(np.nan)
                    source_motp_list.append(np.nan)
                    source_id_switches_list.append(np.nan)
                    continue

                with torch.no_grad():
                    for i, img_path in enumerate(tqdm(image_paths_sf, desc=f" 处理中 {source_folder}")):
                        try:
                            img_pil = Image.open(img_path).convert('RGB')
                        except Exception as e:
                            print(f"\n警告: 跳过文件: {img_path} ({e})")
                            traceback.print_exc()
                            current_preds_sf.append([])
                            continue
                        input_tensor, _, _ = preprocess_image(img_pil, max_size, NORM_MEAN_LIST, NORM_STD_LIST)
                        input_tensor = input_tensor.to(device)
                        try:
                            start_time = time.time()
                            preds = model.predict(input_tensor)
                            torch.cuda.synchronize()
                            elapsed = time.time() - start_time
                            frame_times_sf.append(elapsed)
                            if isinstance(preds, torch.Tensor):
                                preds = preds.cpu().numpy().tolist()
                            elif isinstance(preds, np.ndarray):
                                preds = preds.tolist()

                            filtered_boxes = [box for box in preds if len(box) > 5 and box[5] >= active_conf_threshold]
                            current_preds_sf.append(filtered_boxes)
                        except Exception as e:
                            print(f"\n错误: 处理图像 {image_files[i]} (源 {source_folder}) 失败: {e}")
                            traceback.print_exc()
                            current_preds_sf.append([])
                            continue

                valid_frame_times_sf = [t for t in frame_times_sf if t is not None]
                fps_sf = len(valid_frame_times_sf) / sum(valid_frame_times_sf) if sum(valid_frame_times_sf) > 0 else 0
                map_sf, f1_sf, mota_sf, motp_sf, id_switches_sf = np.nan, np.nan, np.nan, np.nan, np.nan

                if can_compute_gt_metrics:
                    min_len_sf = min(len(current_preds_sf), len(gt_labels_scaled_sf), len(ignore_masks_scaled_sf))
                    if min_len_sf < len(image_paths_sf):
                        print(
                            f"警告: 源 {source_folder} 的预测/GT/忽略列表长度不匹配 ({len(current_preds_sf)}, {len(gt_labels_scaled_sf)}, {len(ignore_masks_scaled_sf)} vs {len(image_paths_sf)} 图像)。仅使用前 {min_len_sf} 帧计算指标。")
                    elif min_len_sf == 0:
                        print(f"警告: 源 {source_folder} 中没有可用于指标计算的帧。")
                    if min_len_sf > 0:
                        try:
                            preds_eval_sf = current_preds_sf[:min_len_sf]
                            gt_eval_sf = gt_labels_scaled_sf[:min_len_sf]
                            ignore_eval_sf = ignore_masks_scaled_sf[:min_len_sf]
                            map_sf = compute_map(preds_eval_sf, gt_eval_sf, ignore_masks=ignore_eval_sf)
                            f1_sf = compute_f1(preds_eval_sf, gt_eval_sf, ignore_masks=ignore_eval_sf)
                            mota_sf, motp_sf, id_switches_sf = compute_mota(preds_eval_sf, gt_eval_sf,
                                                                            ignore_masks=ignore_eval_sf)
                        except Exception as e:
                            print(f"错误: 计算源 {source_folder} 的指标失败: {e}")
                            traceback.print_exc()
                            map_sf, f1_sf, mota_sf, motp_sf, id_switches_sf = np.nan, np.nan, np.nan, np.nan, np.nan
                            print("此源的指标已设为NaN。")
                    else:
                        map_sf, f1_sf, mota_sf, motp_sf, id_switches_sf = np.nan, np.nan, np.nan, np.nan, np.nan

                source_fps_list.append(fps_sf)
                source_map_list.append(map_sf)
                source_f1_list.append(f1_sf)
                source_mota_list.append(mota_sf)
                source_motp_list.append(motp_sf)
                source_id_switches_list.append(id_switches_sf)
                print(
                    f"  {source_folder} 结果: FPS={fps_sf:.2f}, mAP={map_sf:.4f}, F1={f1_sf:.4f}, MOTA={mota_sf:.4f}, MOTP={motp_sf:.4f}, IDSwitches={int(id_switches_sf) if not math.isnan(id_switches_sf) else 'NaN'}")

            overall_avg_fps = np.nanmean(source_fps_list)
            overall_avg_map = np.nanmean(source_map_list)
            overall_avg_f1 = np.nanmean(source_f1_list)
            overall_avg_mota = np.nanmean(source_mota_list)
            overall_avg_motp = np.nanmean(source_motp_list)
            overall_avg_id_switches = np.nanmean(source_id_switches_list)

            combined_results_for_current_tracker.setdefault(result_key, {})
            combined_results_for_current_tracker[result_key][fog_strength] = {
                'fps': overall_avg_fps, 'map': overall_avg_map, 'f1': overall_avg_f1,
                'mota': overall_avg_mota, 'motp': overall_avg_motp, 'id_switches': overall_avg_id_switches
            }
            print(
                f"\n-------- {result_key} (追踪器: {current_tracker_name}, 雾浓度 {fog_strength}) 总体平均结果 (跨源文件夹): --------")
            print(f"  平均 FPS: {overall_avg_fps:.2f}" if not math.isnan(overall_avg_fps) else "  平均 FPS: NaN")
            print(f"  平均 mAP: {overall_avg_map:.4f}" if not math.isnan(overall_avg_map) else "  平均 mAP: NaN")
            print(f"  平均 F1: {overall_avg_f1:.4f}" if not math.isnan(overall_avg_f1) else "  平均 F1: NaN")
            print(f"  平均 MOTA: {overall_avg_mota:.4f}" if not math.isnan(overall_avg_mota) else "  平均 MOTA: NaN")
            print(f"  平均 MOTP: {overall_avg_motp:.4f}" if not math.isnan(overall_avg_motp) else "  平均 MOTP: NaN")
            print(f"  平均 ID 切换次数: {overall_avg_id_switches:.2f}" if not math.isnan(
                overall_avg_id_switches) else "  平均 ID 切换次数: NaN")

        print(f"\n======== 正在为追踪器 {current_tracker_name} 生成图表 (雾浓度: {fog_strength}) ========")
        fog_fps_results = {k: v.get(fog_strength, {}).get('fps', np.nan) for k, v in
                           combined_results_for_current_tracker.items()}
        fog_map_results = {k: v.get(fog_strength, {}).get('map', np.nan) for k, v in
                           combined_results_for_current_tracker.items()}
        fog_f1_results = {k: v.get(fog_strength, {}).get('f1', np.nan) for k, v in
                          combined_results_for_current_tracker.items()}
        fog_mota_results = {k: v.get(fog_strength, {}).get('mota', np.nan) for k, v in
                            combined_results_for_current_tracker.items()}
        fog_motp_results = {k: v.get(fog_strength, {}).get('motp', np.nan) for k, v in
                            combined_results_for_current_tracker.items()}
        fog_id_switches_results = {k: v.get(fog_strength, {}).get('id_switches', np.nan) for k, v in
                                   combined_results_for_current_tracker.items()}

        plot_metric(fog_fps_results, f"各方法平均 FPS (追踪器: {current_tracker_name}, 雾强度: {fog_strength})", "FPS",
                    f"fps_chart_fog_{fog_strength}.png", tracker_fog_result_dir, reverse_sort=True)
        plot_metric(fog_map_results, f"各方法平均 mAP (追踪器: {current_tracker_name}, 雾强度: {fog_strength})", "mAP",
                    f"map_chart_fog_{fog_strength}.png", tracker_fog_result_dir, reverse_sort=True)
        plot_metric(fog_f1_results, f"各方法平均 F1 得分 (追踪器: {current_tracker_name}, 雾强度: {fog_strength})",
                    "F1", f"f1_chart_fog_{fog_strength}.png", tracker_fog_result_dir, reverse_sort=True)
        plot_metric(fog_mota_results, f"各方法平均 MOTA (追踪器: {current_tracker_name}, 雾强度: {fog_strength})",
                    "MOTA", f"mota_chart_fog_{fog_strength}.png", tracker_fog_result_dir, reverse_sort=True)
        plot_metric(fog_motp_results, f"各方法平均 MOTP (追踪器: {current_tracker_name}, 雾强度: {fog_strength})",
                    "MOTP", f"motp_chart_fog_{fog_strength}.png", tracker_fog_result_dir, reverse_sort=False)
        plot_metric(fog_id_switches_results,
                    f"各方法平均 ID 切换次数 (追踪器: {current_tracker_name}, 雾强度: {fog_strength})",
                    "平均 ID 切换次数", f"id_switches_chart_fog_{fog_strength}.png", tracker_fog_result_dir,
                    reverse_sort=False)

    print(f"\n======== 正在为追踪器 {current_tracker_name} 生成汇总评估结果表格 ========")
    csv_path_for_tracker = os.path.join(tracker_base_result_dir, f"{current_tracker_name}_metrics_summary.csv")
    try:
        csv_data = []
        header = ["Method"]
        metrics_order = ['fps', 'map', 'f1', 'mota', 'motp', 'id_switches']
        metric_names_display = {'fps': 'FPS', 'map': 'mAP', 'f1': 'F1 Score', 'mota': 'MOTA', 'motp': 'MOTP',
                                'id_switches': 'Avg ID Switches'}
        for fs in fog_strengths:
            for metric_key in metrics_order:
                header.append(f"{metric_names_display[metric_key]} ({fs})")
        csv_data.append(header)
        method_keys = sorted(list(combined_results_for_current_tracker.keys()))
        for key in method_keys:
            row = [key]
            for fs in fog_strengths:
                results = combined_results_for_current_tracker.get(key, {}).get(fs, {})
                for metric_key in metrics_order:
                    value = results.get(metric_key, np.nan)
                    if metric_key == 'fps':
                        formatted_value = f"{value:.2f}" if not math.isnan(value) else "NaN"
                    elif metric_key in ['map', 'f1', 'mota', 'motp']:
                        formatted_value = f"{value:.4f}" if not math.isnan(value) else "NaN"
                    elif metric_key == 'id_switches':
                        formatted_value = f"{value:.2f}" if not math.isnan(value) else "NaN"
                    else:
                        formatted_value = str(value)
                    row.append(formatted_value)
            csv_data.append(row)
        results_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
        if fog_strengths and f'mAP ({fog_strengths[0]})' in results_df.columns:
            results_df.sort_values(by=f'mAP ({fog_strengths[0]})', ascending=False, inplace=True)
        results_df.to_csv(csv_path_for_tracker, index=False, encoding='utf-8-sig')
        print(f"追踪器 {current_tracker_name} 的汇总评估结果表格已保存到: {csv_path_for_tracker}")
    except Exception as e:
        print(f"错误: 保存追踪器 {current_tracker_name} 的汇总评估结果表格失败: {e}")
        traceback.print_exc()

print("\n所有追踪器的评估和结果保存已完成。")
