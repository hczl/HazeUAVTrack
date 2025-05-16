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

NORM_MEAN_LIST = [0.485, 0.456, 0.406]
NORM_STD_LIST = [0.229, 0.224, 0.225]

source_folders = ['M1007', 'M1008', 'M1009', 'M1010', 'M1011']
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
         sort_value = value if not (isinstance(value, float) and math.isnan(value)) else (-float('inf') if reverse_sort else float('inf'))
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

             plt.text(bar.get_x() + bar.get_width()/2, text_y,
                     label_text, ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
        else:
             plt.text(bar.get_x() + bar.get_width()/2, plt.ylim()[0] + vertical_offset, label_text, ha='center',
                     va='bottom', color='gray', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存到: {save_path}")

dehaze_methods = ["NONE", "DIP", "FALCON", "AOD_NET", "FFA", "AD_NET"]
detector_methods = ["AD_YOLOV11", "DE_NET", "IA_YOLOV3", "YOLOV3", "YOLOV11"]

detector_dehaze_combinations = []
for detector in detector_methods:
    for dehaze in dehaze_methods:
         detector_dehaze_combinations.append((detector, dehaze))

combined_results = {}

for fog_strength in fog_strengths:
    fog_code = f"{int(fog_strength*100):03d}"
    current_base_fog_image_folder = f'{base_fog_image_folder}_{fog_code}'

    print(f"\n======== Processing Fog Strength: {fog_strength} ========")
    print(f"Using base foggy image folder: {current_base_fog_image_folder}")

    fog_result_dir = os.path.join(result_dir, f'fog_{fog_strength}')
    os.makedirs(fog_result_dir, exist_ok=True)

    for detector_name, dehaze_method in detector_dehaze_combinations:
        result_key = f"{detector_name}" if dehaze_method == "NONE" else f"{dehaze_method}_{detector_name}"
        print(f"\n-------- Processing Combination: {result_key} (Fog Strength: {fog_strength}) --------")

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

             model = create_model(cfg)
             device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
             model.to(device)
             model.eval()

        except Exception as e:
             print(f"Error: Failed to create or initialize model {result_key} for fog {fog_strength}: {e}")
             traceback.print_exc()
             combination_load_failed = True
             print("Skipping evaluation for this combination at this fog strength.")

        for source_folder in source_folders:
             print(f"\n---- Processing Source Folder: {source_folder} (Combination: {result_key}, Fog: {fog_strength}) ----")

             image_folder_path = os.path.join(current_base_fog_image_folder, source_folder)
             gt_label_folder_path = os.path.join(base_gt_label_folder, source_folder)
             ignore_mask_folder_path = os.path.join(base_ignore_mask_folder, source_folder)

             if not os.path.exists(image_folder_path):
                  print(f"Warning: Foggy image folder not found: {image_folder_path}. Skipping source {source_folder} for {result_key}.")
                  source_fps_list.append(np.nan)
                  source_map_list.append(np.nan)
                  source_f1_list.append(np.nan)
                  source_mota_list.append(np.nan)
                  source_motp_list.append(np.nan)
                  source_id_switches_list.append(np.nan)
                  continue

             if not os.path.exists(gt_label_folder_path):
                  print(f"Warning: GT label folder not found: {gt_label_folder_path}. Cannot compute GT-dependent metrics for source {source_folder}.")
                  gt_label_folder_path = None
             if not os.path.exists(ignore_mask_folder_path):
                  print(f"Warning: Ignore mask folder not found: {ignore_mask_folder_path}.")
                  ignore_mask_folder_path = None

             image_files = sorted([f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
             image_paths_sf = [os.path.join(image_folder_path, f) for f in image_files]

             if not image_paths_sf:
                  print(f"Warning: No images found in {image_folder_path}. Skipping source {source_folder} for {result_key}.")
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
                     _, orig_size_sf, resized_size_sf = preprocess_image(first_img_pil_sf, max_size, NORM_MEAN_LIST, NORM_STD_LIST)

                     gt_labels_sf, ignore_masks_sf = load_annotations(gt_label_folder_path, ignore_mask_folder_path, len(image_paths_sf))

                     gt_labels_scaled_sf = scale_ground_truth_boxes(gt_labels_sf, orig_size_sf, resized_size_sf)
                     ignore_masks_scaled_sf = scale_ignore_regions(ignore_masks_sf, orig_size_sf, resized_size_sf)

                     if len(gt_labels_scaled_sf) == len(image_paths_sf) and len(ignore_masks_scaled_sf) == len(image_paths_sf):
                          can_compute_gt_metrics = True
                          print(f"Successfully loaded and scaled GT/Ignores for source {source_folder}.")
                     else:
                          print(f"Warning: GT/Ignore count mismatch for {source_folder} ({len(gt_labels_scaled_sf)}/{len(ignore_masks_scaled_sf)} vs {len(image_paths_sf)} images). Cannot compute GT-dependent metrics for this source.")
                          gt_labels_scaled_sf = []
                          ignore_masks_scaled_sf = []

                 except Exception as e:
                     print(f"Error: Loading/Scaling GT/Ignores failed for source {source_folder}: {e}")
                     traceback.print_exc()
                     gt_labels_scaled_sf = []
                     ignore_masks_scaled_sf = []
                     print("Cannot compute GT-dependent metrics for this source folder.")

             elif gt_label_folder_path is None or ignore_mask_folder_path is None:
                  print(f"Skipping GT-dependent metrics for source {source_folder} due to missing GT/Ignore folders.")

             frame_times_sf = []
             current_preds_sf = []

             if combination_load_failed:
                 print(f"Skipping image processing for source {source_folder} due to model loading failure.")
                 source_fps_list.append(np.nan)
                 source_map_list.append(np.nan)
                 source_f1_list.append(np.nan)
                 source_mota_list.append(np.nan)
                 source_motp_list.append(np.nan)
                 source_id_switches_list.append(np.nan)
                 continue

             with torch.no_grad():
                 for i, img_path in enumerate(tqdm(image_paths_sf, desc=f" Processing {source_folder}")):
                     try:
                         img_pil = Image.open(img_path).convert('RGB')
                     except Exception as e:
                         print(f"\nWarning: Skipping file: {img_path} ({e})")
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

                         conf_threshold = 0.86
                         filtered_boxes = [box for box in preds if len(box) > 5 and box[5] >= conf_threshold]

                         current_preds_sf.append(filtered_boxes)

                     except Exception as e:
                         print(f"\nError: Processing image {image_files[i]} in {source_folder} failed: {e}")
                         traceback.print_exc()
                         current_preds_sf.append([])
                         continue

             valid_frame_times_sf = [t for t in frame_times_sf if t is not None]
             fps_sf = len(valid_frame_times_sf) / sum(valid_frame_times_sf) if sum(valid_frame_times_sf) > 0 else 0

             map_sf = np.nan
             f1_sf = np.nan
             mota_sf = np.nan
             motp_sf = np.nan
             id_switches_sf = np.nan

             if can_compute_gt_metrics:
                  min_len_sf = min(len(current_preds_sf), len(gt_labels_scaled_sf), len(ignore_masks_scaled_sf))
                  if min_len_sf < len(image_paths_sf):
                       print(f"Warning: Preds/GT/Ignore list length mismatch for source {source_folder} ({len(current_preds_sf)}, {len(gt_labels_scaled_sf)}, {len(ignore_masks_scaled_sf)} vs {len(image_paths_sf)} images). Only using first {min_len_sf} frames for metrics.")
                  elif min_len_sf == 0:
                       print(f"Warning: No frames available for metric calculation in source {source_folder}.")

                  if min_len_sf > 0:
                       try:
                           preds_eval_sf = current_preds_sf[:min_len_sf]
                           gt_eval_sf = gt_labels_scaled_sf[:min_len_sf]
                           ignore_eval_sf = ignore_masks_scaled_sf[:min_len_sf]

                           map_sf = compute_map(preds_eval_sf, gt_eval_sf, ignore_masks=ignore_eval_sf)
                           f1_sf = compute_f1(preds_eval_sf, gt_eval_sf, ignore_masks=ignore_eval_sf)
                           mota_sf, motp_sf, id_switches_sf = compute_mota(preds_eval_sf, gt_eval_sf, ignore_masks=ignore_eval_sf)

                       except Exception as e:
                           print(f"Error: Computing metrics for source {source_folder} failed: {e}")
                           traceback.print_exc()
                           map_sf = np.nan
                           f1_sf = np.nan
                           mota_sf = np.nan
                           motp_sf = np.nan
                           id_switches_sf = np.nan
                           print("Metrics for this source set to NaN.")
                  else:
                      map_sf = np.nan
                      f1_sf = np.nan
                      mota_sf = np.nan
                      motp_sf = np.nan
                      id_switches_sf = np.nan

             source_fps_list.append(fps_sf)
             source_map_list.append(map_sf)
             source_f1_list.append(f1_sf)
             source_mota_list.append(mota_sf)
             source_motp_list.append(motp_sf)
             source_id_switches_list.append(id_switches_sf)

             print(f"  {source_folder} results: FPS={fps_sf:.2f}, mAP={map_sf:.4f}, F1={f1_sf:.4f}, MOTA={mota_sf:.4f}, MOTP={motp_sf:.4f}, IDSwitches={int(id_switches_sf) if not math.isnan(id_switches_sf) else 'NaN'}")

        overall_avg_fps = np.nanmean(source_fps_list)
        overall_avg_map = np.nanmean(source_map_list)
        overall_avg_f1 = np.nanmean(source_f1_list)
        overall_avg_mota = np.nanmean(source_mota_list)
        overall_avg_motp = np.nanmean(source_motp_list)
        overall_avg_id_switches = np.nanmean(source_id_switches_list)

        combined_results.setdefault(result_key, {})
        combined_results[result_key][fog_strength] = {
            'fps': overall_avg_fps,
            'map': overall_avg_map,
            'f1': overall_avg_f1,
            'mota': overall_avg_mota,
            'motp': overall_avg_motp,
            'id_switches': overall_avg_id_switches
        }

        print(f"\n-------- {result_key} (Fog Strength {fog_strength}) Overall Average Results (Across Source Folders): --------")
        print(f"  Avg FPS: {overall_avg_fps:.2f}" if not math.isnan(overall_avg_fps) else "  Avg FPS: NaN")
        print(f"  Avg mAP: {overall_avg_map:.4f}" if not math.isnan(overall_avg_map) else "  Avg mAP: NaN")
        print(f"  Avg F1: {overall_avg_f1:.4f}" if not math.isnan(overall_avg_f1) else "  Avg F1: NaN")
        print(f"  Avg MOTA: {overall_avg_mota:.4f}" if not math.isnan(overall_avg_mota) else "  Avg MOTA: NaN")
        print(f"  Avg MOTP: {overall_avg_motp:.4f}" if not math.isnan(overall_avg_motp) else "  Avg MOTP: NaN")
        print(f"  Avg ID Switches: {overall_avg_id_switches:.2f}" if not math.isnan(overall_avg_id_switches) else "  Avg ID Switches: NaN")

    print(f"\n======== Generating plots for Fog Strength: {fog_strength} ========")

    fog_fps_results = {k: v.get(fog_strength, {}).get('fps', np.nan) for k, v in combined_results.items()}
    fog_map_results = {k: v.get(fog_strength, {}).get('map', np.nan) for k, v in combined_results.items()}
    fog_f1_results = {k: v.get(fog_strength, {}).get('f1', np.nan) for k, v in combined_results.items()}
    fog_mota_results = {k: v.get(fog_strength, {}).get('mota', np.nan) for k, v in combined_results.items()}
    fog_motp_results = {k: v.get(fog_strength, {}).get('motp', np.nan) for k, v in combined_results.items()}
    fog_id_switches_results = {k: v.get(fog_strength, {}).get('id_switches', np.nan) for k, v in combined_results.items()}

    plot_metric(fog_fps_results, f"各方法平均 FPS (雾强度: {fog_strength})", "FPS", f"fps_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=True)
    plot_metric(fog_map_results, f"各方法平均 mAP (雾强度: {fog_strength})", "mAP", f"map_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=True)
    plot_metric(fog_f1_results, f"各方法平均 F1 得分 (雾强度: {fog_strength})", "F1", f"f1_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=True)
    plot_metric(fog_mota_results, f"各方法平均 MOTA (雾强度: {fog_strength})", "MOTA", f"mota_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=True)
    plot_metric(fog_motp_results, f"各方法平均 MOTP (雾强度: {fog_strength})", "MOTP", f"motp_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=False)
    plot_metric(fog_id_switches_results, f"各方法平均 ID 切换次数 (雾强度: {fog_strength})", "平均 ID 切换次数", f"id_switches_chart_fog_{fog_strength}.png", fog_result_dir, reverse_sort=False)

print("\n======== Generating Combined Evaluation Summary Table ========")

csv_path = os.path.join(result_dir, "combined_detector_metrics_summary.csv")

try:
    csv_data = []
    header = ["Method"]
    metrics_order = ['fps', 'map', 'f1', 'mota', 'motp', 'id_switches']
    metric_names_display = {'fps': 'FPS', 'map': 'mAP', 'f1': 'F1 Score', 'mota': 'MOTA', 'motp': 'MOTP', 'id_switches': 'Avg ID Switches'}

    for fs in fog_strengths:
        for metric_key in metrics_order:
             header.append(f"{metric_names_display[metric_key]} ({fs})")
    csv_data.append(header)

    method_keys = sorted(list(combined_results.keys()))

    for key in method_keys:
        row = [key]
        for fs in fog_strengths:
            results = combined_results.get(key, {}).get(fs, {})

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

    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"汇总评估结果表格已保存到: {csv_path}")

except Exception as e:
    print(f"错误: 保存汇总评估结果表格失败: {e}")
    traceback.print_exc()

print("\n所有评估和结果保存已完成。")
