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
import traceback
import csv

# 设置 Matplotlib 字体，以支持中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 设置 PyTorch 模型下载目录
os.environ['TORCH_HOME'] = './.torch'

# 雾强度的列表
# 示例雾强度，需要确保对应的数据文件夹存在
fog_strengths = [0.5, 0.75]

# 数据集文件夹列表 (地面真实和对应的雾图都在这些文件夹下)
gt_source_folders = ['M1005', 'M0301', 'M1002', 'M1202', 'M0205', 'M1007']
# 假设地面真实文件夹路径的基础部分
base_gt_folder = 'data/UAV-M/UAV-benchmark-M'
# 假设雾图文件夹路径的基础部分 (需要根据雾强度调整后缀)
base_fog_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog' # Will append _XXX

# --- 配置参数 ---
max_size = 1024
result_dir = 'result/dehaze'
video_dir = 'result/video'

# 创建基本的结果目录
os.makedirs(result_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# 图像预处理
transform_to_tensor = transforms.ToTensor()

# 标准化参数 (ImageNet)
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# 定义去雾方法列表
dehazes = ['NONE', 'DIP', 'FALCON', 'AOD_NET', 'FFA', 'AD_NET']

# YAML 配置文件路径 ( assumed to be general and method-specific details loaded later)
yaml_path = 'configs/DIP.yaml' # 注意：这个路径用于加载初始配置结构。
# 实际的方法 ('dehaze') 和雾强度 (fog_strength) 会动态设置。

# 存储所有结果的字典
# 结构: { method_name: { fog_strength: { 'fps': avg_over_sources, 'psnr': avg_over_sources, 'ssim': avg_over_sources } } }
combined_results = {}

# 遍历不同的雾强度
for fog_strength in fog_strengths:
    # 根据雾强度构建当前的雾图基础文件夹路径
    fog_code = f"{int(fog_strength*100):03d}"
    current_base_fog_folder = f'{base_fog_folder}_{fog_code}'

    print(f"\n======== 处理雾强度: {fog_strength} ========")
    print(f"使用雾图基础文件夹: {current_base_fog_folder}")

    # 创建每个雾强度的结果目录 (用于存放 per-fog plots 和 videos)
    fog_result_dir = os.path.join(result_dir, f'fog_{fog_strength}')
    video_result_dir = os.path.join(video_dir, f'fog_{fog_strength}')
    os.makedirs(fog_result_dir, exist_ok=True)
    os.makedirs(video_result_dir, exist_ok=True)

    # 遍历不同的去雾方法
    for dehaze in dehazes:
        print(f"\n-------- 处理去雾方法: {dehaze} (雾强度: {fog_strength}) --------")

        model = None
        cfg = None
        device = "cpu"
        is_model_loaded = False
        method_load_failed = False # Flag to skip processing for this method/fog if model fails

        # 初始化 combined_results 中该方法的条目（如果不存在）
        combined_results.setdefault(dehaze, {})

        # 加载模型 (如果不是 'NONE')
        if dehaze != 'NONE':
            try:
                cfg = load_config(yaml_path) # 加载基础配置
                cfg['method']['dehaze'] = dehaze # 设置具体方法
                # cfg['dataset']['fog_strength'] = fog_strength  # 可选：在配置中设置雾强度，如果模型依赖此参数
                model = create_model(cfg)
                device = cfg['device'] if torch.cuda.is_available() else "cpu"
                model.to(device)

                # 尝试加载模型权重
                try:
                    model.load_model()
                    is_model_loaded = True
                    print(f"成功加载模型权重 for {dehaze}.")
                except Exception as e:
                    print(f"警告: 加载模型权重失败 for {dehaze} (雾强度 {fog_strength}): {e}")
                    print("将使用未加载权重的模型。")
                    traceback.print_exc()

                model.eval()
            except Exception as e:
                print(f"错误: 创建或加载模型 {dehaze} 失败 for 雾强度 {fog_strength}: {e}")
                traceback.print_exc()
                method_load_failed = True # Mark as failed so we skip processing images

        # 存储该方法在该雾强度下，每个数据集文件夹的评估结果
        # { source_folder_name: { 'fps': avg_sf, 'psnr': avg_sf, 'ssim': avg_sf } }
        source_folder_summary_metrics = {}
        processed_frames_for_video = [] # 收集所有数据集文件夹的帧用于生成一个视频

        # 遍历不同的数据集文件夹
        for source_folder in gt_source_folders:
            print(f"\n---- 处理数据集文件夹: {source_folder} (方法: {dehaze}, 雾强度: {fog_strength}) ----")

            gt_folder_path = os.path.join(base_gt_folder, source_folder)
            current_image_folder_path = os.path.join(current_base_fog_folder, source_folder)

            # 检查当前数据集文件夹和对应的雾图文件夹是否存在
            if not os.path.exists(current_image_folder_path) or not os.path.exists(gt_folder_path):
                print(f"警告: 数据集文件夹或雾图文件夹不存在或不完整:")
                print(f"  GT 路径: {gt_folder_path}")
                print(f"  雾图路径: {current_image_folder_path}")
                print(f"跳过文件夹 {source_folder} 的处理 for {dehaze} (雾强度 {fog_strength}).")
                source_folder_summary_metrics[source_folder] = {'fps': np.nan, 'psnr': np.nan, 'ssim': np.nan}
                continue # 跳到下一个数据集文件夹

            image_files_for_current_source = sorted([
                f for f in os.listdir(current_image_folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            if not image_files_for_current_source:
                print(f"警告: 数据集文件夹 {current_image_folder_path} 中没有找到图片文件。")
                print(f"跳过文件夹 {source_folder} 的处理 for {dehaze} (雾强度 {fog_strength}).")
                source_folder_summary_metrics[source_folder] = {'fps': np.nan, 'psnr': np.nan, 'ssim': np.nan}
                continue # 跳到下一个数据集文件夹

            print(f"找到 {len(image_files_for_current_source)} 张图片在 {source_folder} 中进行评估.")

            # Lists to collect metrics for THIS source folder
            frame_times_sf = []
            psnr_list_sf = []
            ssim_list_sf = []

            # 开始处理当前数据集文件夹下的图片
            with torch.no_grad():
                for i, file_name in enumerate(image_files_for_current_source):
                    # 构建当前雾图和地面真实图片路径
                    image_path = os.path.join(current_image_folder_path, file_name)
                    gt_path = os.path.join(gt_folder_path, file_name)

                    try:
                        # 打开并转换为 RGB
                        image_pil = Image.open(image_path).convert("RGB")
                        gt_image_pil = Image.open(gt_path).convert("RGB")
                    except Exception as e:
                        print(f"警告: 跳过文件 {file_name} (无法打开): {e}")
                        traceback.print_exc()
                        continue

                    image_tensor = transform_to_tensor(image_pil)
                    gt_tensor = transform_to_tensor(gt_image_pil)

                    # 调整图片大小以进行处理（保持纵横比，四舍五入到 32 的倍数）
                    orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
                    r = min(1.0, max_size / float(max(orig_w, orig_h)))
                    new_h = max(32, int(round(orig_h * r / 32) * 32))
                    new_w = max(32, int(round(orig_w * r / 32) * 32))

                    try:
                         # 使用 BICUBIC 插值进行调整
                        image_resized = F.resize(image_tensor, (new_h, new_w),
                                                 interpolation=transforms.InterpolationMode.BICUBIC)
                        gt_resized = F.resize(gt_tensor, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
                    except Exception as e:
                        print(f"警告: 跳过文件 {file_name} (resize 失败): {e}")
                        traceback.print_exc()
                        continue

                    start_time = time.time()
                    processed_image_for_metrics = None # 用于 PSNR/SSIM 计算的图片张量 [C, H, W] (0-1 范围)
                    current_frame_np = None # 用于视频的 numpy 帧

                    if dehaze == 'NONE' or method_load_failed:
                        # 如果是 'NONE' 方法或模型加载失败，则使用原始（调整大小后）的雾图
                        processed_image_for_metrics = image_resized
                        current_frame_np = (image_resized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        # 确保 CUDA 同步，即使 GPU 未被 'NONE' 使用
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elapsed = time.time() - start_time
                        frame_times_sf.append(elapsed) # 计入计时

                    else: # 使用去雾模型处理
                        try:
                            # 对输入进行标准化（针对模型输入要求）
                            normalized_input_tensor = F.normalize(image_resized, mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                            input_tensor = normalized_input_tensor.unsqueeze(0).to(device)

                            # 执行预测
                            output = model.predict(input_tensor)

                            if not isinstance(output, torch.Tensor):
                                print(f"警告: 模型预测输出不是 Tensor for {file_name}. Skipping.")
                                psnr_list_sf.append(np.nan)
                                ssim_list_sf.append(np.nan)
                                continue # 跳过该文件的指标计算和视频帧

                            # 反标准化并限制输出范围
                            # 假设模型的输出已经在 [0, 1] 范围内
                            # 如果你的模型输出的是经过标准化的数据，你可能需要：
                            output_denorm = output * NORM_STD.to(output.device) + NORM_MEAN.to(output.device)
                            processed_image_for_metrics = torch.clamp(output_denorm, 0, 1).squeeze(0).cpu()
                            # 如果模型输出直接是 [0,1]：
                            # processed_image_for_metrics = torch.clamp(output, 0, 1).squeeze(0).cpu()

                            current_frame_np = (processed_image_for_metrics.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                            # 同步并记录时间
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            elapsed = time.time() - start_time
                            frame_times_sf.append(elapsed) # 计入计时

                        except Exception as e:
                            print(f"警告: 跳过文件 {file_name} (模型预测或后处理失败): {e}")
                            traceback.print_exc()
                            psnr_list_sf.append(np.nan)
                            ssim_list_sf.append(np.nan)
                            # 如果处理失败，不计入计时
                            continue # 跳过该文件的指标计算和视频帧

                    # 计算 PSNR 和 SSIM，如果 processed_image_for_metrics 可用
                    if processed_image_for_metrics is not None:
                        try:
                            # 确保 GT 也是 float [0,1] 用于指标计算
                            gt_np = gt_resized.permute(1, 2, 0).numpy() # 已经是 float [0,1]
                            processed_np_float = processed_image_for_metrics.permute(1, 2, 0).numpy() # 已经是 float [0,1]

                            psnr_val = psnr(gt_np, processed_np_float, data_range=1)
                            ssim_val = ssim(gt_np, processed_np_float, channel_axis=2, data_range=1)

                            psnr_list_sf.append(psnr_val)
                            ssim_list_sf.append(ssim_val)
                        except Exception as e:
                            print(f"警告: 计算 PSNR/SSIM 失败 for {file_name}: {e}")
                            traceback.print_exc()
                            psnr_list_sf.append(np.nan)
                            ssim_list_sf.append(np.nan)

                    # 添加当前帧到视频帧列表
                    if current_frame_np is not None:
                         processed_frames_for_video.append(current_frame_np)


                    # 每处理一定数量或所有图片后打印进度
                    if (i + 1) % 50 == 0 or (i + 1) == len(image_files_for_current_source):
                        print(f"  已处理 {i + 1}/{len(image_files_for_current_source)} 张图片在 {source_folder} 中...")


            # --- 计算当前方法、雾强度和数据集文件夹下的平均指标 ---
            valid_psnr_sf = [x for x in psnr_list_sf if not (isinstance(x, float) and math.isnan(x))]
            valid_ssim_sf = [x for x in ssim_list_sf if not (isinstance(x, float) and math.isnan(x))]

            total_time_sf = sum(frame_times_sf)
            num_processed_frames_for_fps_sf = len(frame_times_sf) # 成功计时的帧数
            fps_sf = num_processed_frames_for_fps_sf / total_time_sf if total_time_sf > 0 else 0

            avg_psnr_sf = np.mean(valid_psnr_sf) if valid_psnr_sf else np.nan
            avg_ssim_sf = np.mean(valid_ssim_sf) if valid_ssim_sf else np.nan

            # 将该数据集文件夹的结果存储到临时字典中
            source_folder_summary_metrics[source_folder] = {
                'fps': fps_sf,
                'psnr': avg_psnr_sf,
                'ssim': avg_ssim_sf
            }

            print(f"\n{dehaze} (雾强度 {fog_strength}, 数据集 {source_folder}) 评估结果:")
            print(f"  总处理时间: {total_time_sf:.2f} 秒")
            print(f"  成功计时帧数: {num_processed_frames_for_fps_sf}")
            print(f"  平均 FPS: {fps_sf:.2f}")
            print(f"  有效 PSNR 帧数: {len(valid_psnr_sf)}")
            print(f"  平均 PSNR: {avg_psnr_sf:.2f}" if not math.isnan(avg_psnr_sf) else "  平均 PSNR: NaN")
            print(f"  有效 SSIM 帧数: {len(valid_ssim_sf)}")
            f_ssim_str_sf = f"{avg_ssim_sf:.3f}" if not math.isnan(avg_ssim_sf) else "NaN"
            print(f"  平均 SSIM: {f_ssim_str_sf}")

        # --- 完成当前方法和雾强度下所有数据集文件夹的处理 ---

        # 计算该方法在该雾强度下，所有数据集文件夹的平均指标
        all_fps_sf = [res['fps'] for res in source_folder_summary_metrics.values()]
        all_psnr_sf = [res['psnr'] for res in source_folder_summary_metrics.values()]
        all_ssim_sf = [res['ssim'] for res in source_folder_summary_metrics.values()]

        # 使用 nanmean 来忽略 NaN 值进行平均计算
        overall_avg_fps = np.nanmean(all_fps_sf)
        overall_avg_psnr = np.nanmean(all_psnr_sf)
        overall_avg_ssim = np.nanmean(all_ssim_sf)

        # 将最终的整体平均结果存储到 combined_results 字典中
        combined_results[dehaze][fog_strength] = {
            'fps': overall_avg_fps,
            'psnr': overall_avg_psnr,
            'ssim': overall_avg_ssim
        }

        print(f"\n-------- {dehaze} (雾强度 {fog_strength}) 汇总平均结果 (跨数据集文件夹): --------")
        print(f"  平均 FPS: {overall_avg_fps:.2f}" if not math.isnan(overall_avg_fps) else "  平均 FPS: NaN")
        print(f"  平均 PSNR: {overall_avg_psnr:.2f}" if not math.isnan(overall_avg_psnr) else "  平均 PSNR: NaN")
        print(f"  平均 SSIM: {overall_avg_ssim:.3f}" if not math.isnan(overall_avg_ssim) else "  平均 SSIM: NaN")


        # 为当前方法和雾强度生成视频 (包含所有数据集文件夹的帧)
        if dehaze != 'NONE' and processed_frames_for_video:
            print(f"正在生成视频: {dehaze}_fog_{fog_strength}.mp4")
            # 视频保存在对应雾强度的子目录下
            video_path = os.path.join(video_result_dir, f'{dehaze}.mp4')
            # 确保视频帧大小一致 (使用第一帧的大小)
            height, width, _ = processed_frames_for_video[0].shape if processed_frames_for_video else (0,0,0)
            if height == 0 or width == 0:
                print(f"警告: 没有有效帧生成视频 for {dehaze} (雾强度 {fog_strength}).")
            else:
                # 视频尺寸必须是偶数
                width = width if width % 2 == 0 else width - (width % 2)
                height = height if height % 2 == 0 else height - (height % 2)

                try:
                    # 使用常用的编解码器，如 MP4V
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # 假设输出视频是 30 FPS
                    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

                    for frame in processed_frames_for_video:
                         # 如果需要，调整帧大小
                        if frame.shape[0] != height or frame.shape[1] != width:
                            frame = cv2.resize(frame, (width, height))
                        # OpenCV 默认 BGR，PIL/numpy 默认 RGB
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)

                    out.release()
                    print(f"视频已保存到: {video_path}")
                except Exception as e:
                    print(f"错误: 生成视频 {video_path} 失败: {e}")
                    traceback.print_exc()

        elif dehaze != 'NONE':
            print(f"没有成功处理的帧，跳过视频生成 for {dehaze} (雾强度 {fog_strength})。")


    # --- 完成当前雾强度下所有方法的处理 ---

    # --- 为当前雾强度生成图表 (使用跨数据集文件夹的平均结果) ---
    print(f"\n======== 生成雾强度 {fog_strength} 的评估结果图表 (跨数据集文件夹平均) ========")

    def plot_metric(data_dict, title, ylabel, filename, is_integer=False, reverse_sort=True, include_none=True):
        # 准备绘图数据
        plot_items = []
        for name, value in data_dict.items():
             if not include_none and name == 'NONE':
                 continue
             # 使用值进行排序，NaN 放在末尾
             sort_value = value if not (isinstance(value, float) and math.isnan(value)) else (-float('inf') if reverse_sort else float('inf'))
             plot_items.append((sort_value, name, value))

        # 按值排序，然后按名称字母顺序排序（用于处理并列情况）
        sorted_items = sorted(plot_items, key=lambda item: (item[0], item[1]), reverse=reverse_sort)

        names = [item[1] for item in sorted_items]
        values = [item[2] for item in sorted_items]

        # 如果没有有效数据，则不绘制
        valid_values_for_plot = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
        if not valid_values_for_plot:
            print(f"没有有效数据可绘制 {title} for fog {fog_strength}。")
            return

        plt.figure(figsize=(10, 7))
        bars = plt.bar(names, values)
        plt.title(title) # 标题中已包含雾强度
        plt.ylabel(ylabel)

        plt.xticks(rotation=45, ha='right')

        # 根据值的范围动态调整 Y 轴限制
        if valid_values_for_plot:
            min_val = min(valid_values_for_plot)
            max_val = max(valid_values_for_plot)
            # 为 Y 轴添加一些填充
            padding = (max_val - min_val) * 0.1 if max_val != min_val else abs(max_val) * 0.1 if max_val != 0 else 0.1
            lower_bound = min(0, min_val - padding)
            upper_bound = max(0.1, max_val + padding) # 确保上限至少为 0.1

            # 特殊情况：所有有效值都相同
            if min_val == max_val:
                 lower_bound = min_val * 0.9 if min_val != 0 else -0.1
                 upper_bound = min_val * 1.1 if min_val != 0 else 0.1

            plt.ylim(lower_bound, upper_bound)
        else:
            plt.ylim(0, 1) # 没有有效数据时的默认限制

        # 在柱状图顶部添加数值标签
        y_range = plt.ylim()[1] - plt.ylim()[0]
        vertical_offset = y_range * 0.02 if y_range > 0 else 0.02 # 标签位置的小偏移量

        for bar in bars:
            h = bar.get_height()
            if math.isnan(h):
                label_text = "NaN"
            elif is_integer:
                label_text = f"{int(h)}"
            else:
                # 对于接近 0 的小值，使用更多小数位
                label_text = f"{h:.2f}" if abs(h) >= 0.01 or h == 0 else f"{h:.4f}"

            # 根据高度正负将标签放在柱状图上方或下方
            if not math.isnan(h):
                 # Adjust text position slightly for clarity
                 text_y = h + (vertical_offset if h >= 0 else -vertical_offset)
                 # Ensure text is within plot bounds (basic check)
                 text_y = max(plt.ylim()[0], min(plt.ylim()[1], text_y))

                 plt.text(bar.get_x() + bar.get_width() / 2, text_y,
                         label_text, ha='center', va='bottom' if h >= 0 else 'top')
            else:
                 # 如果是 NaN，将标签放在 Y 轴底部附近
                 plt.text(bar.get_x() + bar.get_width() / 2, plt.ylim()[0] + vertical_offset, label_text, ha='center',
                         va='bottom', color='gray')


        plt.tight_layout() # 调整布局以防止标签重叠
        # 保存图表，文件名包含雾强度
        save_path = os.path.join(fog_result_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"图表已保存到: {save_path}")

    # 提取当前雾强度下各方法的平均结果用于绘图
    current_fog_avg_results = {}
    for dehaze_method in dehazes:
        # combined_results[dehaze_method][fog_strength] 存储的是该方法在该雾强度下的整体平均结果
        current_fog_avg_results[dehaze_method] = combined_results.get(dehaze_method, {}).get(fog_strength, {})


    # 使用提取的平均结果为当前雾强度生成图表
    # FPS: 越高越好
    plot_metric({m: r.get('fps', np.nan) for m, r in current_fog_avg_results.items()},
                f"各去雾方法的平均 FPS (雾强度: {fog_strength})", "FPS", "fps_chart.png",
                reverse_sort=True, include_none=True)
    # PSNR: 越高越好
    plot_metric({m: r.get('psnr', np.nan) for m, r in current_fog_avg_results.items()},
                f"各去雾方法的平均 PSNR (雾强度: {fog_strength})", "PSNR", "psnr_chart.png",
                reverse_sort=True, include_none=True)
    # SSIM: 越高越好
    plot_metric({m: r.get('ssim', np.nan) for m, r in current_fog_avg_results.items()},
                f"各去雾方法的平均 SSIM (雾强度: {fog_strength})", "SSIM", "ssim_chart.png",
                reverse_sort=True, include_none=True)


# --- 处理完所有雾强度后，写入汇总的 CSV 文件 (使用最终的整体平均结果) ---

print("\n======== 生成汇总评估结果表格 ========")

combined_table_filename = os.path.join(result_dir, 'combined_evaluation_results_avg_across_sources.csv')

try:
    with open(combined_table_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 创建表头行: 'Method', 'FPS (雾强度1)', 'PSNR (雾强度1)', 'SSIM (雾强度1)', 'FPS (雾强度2)', ...
        header = ['Method']
        for fs in fog_strengths:
            header.append(f'FPS ({fs})')
            header.append(f'PSNR ({fs})')
            header.append(f'SSIM ({fs})')
        writer.writerow(header)

        # 写入每个方法的数据行
        # 确保按照 'dehazes' 列表中的顺序写入方法
        for dehaze_method in dehazes:
            row_data = [dehaze_method]
            # 获取该方法在所有雾强度下的最终平均结果
            method_results = combined_results.get(dehaze_method, {})

            for fs in fog_strengths:
                # 获取该特定雾强度的最终平均结果，如果找不到则默认为空字典
                results = method_results.get(fs, {})

                # 获取指标值，如果找不到则默认为 NaN
                fps_val = results.get('fps', np.nan)
                psnr_val = results.get('psnr', np.nan)
                ssim_val = results.get('ssim', np.nan)

                # 格式化 CSV 中的值（处理 NaN）
                fps_str = f"{fps_val:.2f}" if not math.isnan(fps_val) else "NaN"
                psnr_str = f"{psnr_val:.2f}" if not math.isnan(psnr_val) else "NaN"
                ssim_str = f"{ssim_val:.3f}" if not math.isnan(ssim_val) else "NaN"

                row_data.append(fps_str)
                row_data.append(psnr_str)
                row_data.append(ssim_str)

            writer.writerow(row_data)

    print(f"汇总评估结果表格已保存到: {combined_table_filename}")

except Exception as e:
    print(f"错误: 保存汇总评估结果表格失败: {e}")
    traceback.print_exc()

print("\n所有评估和结果保存已完成。")
