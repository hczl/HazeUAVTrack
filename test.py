import math
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F # 导入 functional 模块

number = "1101"
# 读取标签数据
data = pd.read_csv(f'data/UAV-M/UAV-benchmark-MOTD_v1.0/GT/M{number}_gt_whole.txt', header=None,
                   names=['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                          'out_of_view', 'occlusion', 'object_category'])
data1 = pd.read_csv(f'data/UAV-M/UAV-benchmark-MOTD_v1.0/GT/M{number}_gt_ignore.txt', header=None,
                   names=['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                          'out_of_view', 'occlusion', 'object_category'])
# 图像文件夹路径
image_folder = f'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M{number}'

# 获取所有图像文件名并排序，确保按照文件名中的数字部分排序
image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split('img')[1].split('.')[0]))

# 设置目标输出尺寸 (与你计算的 new_w, new_h 保持一致)
# 注意：视频写入器需要固定的尺寸，这个尺寸应该是你resize后的目标尺寸
# 或者，你可以根据第一帧的resize结果动态确定
# 这里我们假设你希望输出视频的分辨率就是resize后的 new_w, new_h
# 如果你想输出固定尺寸的视频，即使图片被resize了，那么你需要将resized的图片再resize到frame_width, frame_height
# 假设你希望输出视频尺寸就是resize后的尺寸：
# 找到第一帧来确定resize后的尺寸
first_img_path = os.path.join(image_folder, image_files[0])
frame_original = cv2.imread(first_img_path)
o_h_first, o_w_first, _ = frame_original.shape

max_size = 640
r_first = min(1.0, max_size / o_w_first)
r_first = min(r_first, max_size / o_h_first)
new_h_first = int(round(o_h_first * r_first))
new_w_first = int(round(o_w_first * r_first))
new_h_first = max(32, math.floor(new_h_first / 32) * 32)
new_w_first = max(32, math.floor(new_w_first / 32) * 32)

frame_width = new_w_first  # 使用resize后的宽度作为视频宽度
frame_height = new_h_first # 使用resize后的高度作为视频高度

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

# 遍历每一帧图片
for image_file in image_files:
    # 获取当前帧图像路径
    img_path = os.path.join(image_folder, image_file)

    # 读取图像帧
    frame_original_np = cv2.imread(img_path) # 使用新的变量名表示原始numpy数组

    # 获取原始尺寸
    o_h, o_w, _ = frame_original_np.shape

    # 获取当前帧的索引
    frame_index = int(image_file.split('img')[1].split('.')[0])  # 提取文件名中的数字部分

    # 获取当前帧的标签数据
    frame_data = data[data['frame_index'] == frame_index]
    frame_data1 = data1[data1['frame_index'] == frame_index]

    # --- Resize Logic ---
    max_size = 640
    r = min(1.0, max_size / o_w)
    r = min(r, max_size / o_h)
    new_h = int(round(o_h * r))
    new_w = int(round(o_w * r))
    new_h = max(32, math.floor(new_h / 32) * 32)
    new_w = max(32, math.floor(new_w / 32) * 32)

    # Recalculate scale factors based on the *final* new_h, new_w
    scale_factor_w_final = new_w / o_w
    scale_factor_h_final = new_h / o_h

    # Convert to PIL, Resize, Convert back to NumPy
    frame_pil = Image.fromarray(frame_original_np) # Convert original numpy to PIL
    frame_resized_pil = F.resize(frame_pil, (new_h, new_w)) # Resize PIL (result is PIL)
    frame_for_drawing_np = np.array(frame_resized_pil) # Convert resized PIL back to numpy array
    # --- End Resize Logic ---

    # 现在所有的OpenCV绘图和写入操作都应该使用 frame_for_drawing_np

    # 遍历该帧的所有目标 (frame_data)
    for _, row in frame_data.iterrows():
        # 提取边界框信息 (原始坐标)
        x_orig, y_orig, w_orig, h_orig = int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])

        # 缩放边界框坐标到新的尺寸
        x_scaled = int(x_orig * scale_factor_w_final)
        y_scaled = int(y_orig * scale_factor_h_final)
        w_scaled = int(w_orig * scale_factor_w_final)
        h_scaled = int(h_orig * scale_factor_h_final)

        # 获取目标ID作为标签
        label = f"ID: {row['target_id']}"

        # 绘制边界框在缩放后的图像上
        # 使用 frame_for_drawing_np
        cv2.rectangle(frame_for_drawing_np, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)  # 使用绿色框

        # 在左上角添加标签
        # 使用 frame_for_drawing_np
        cv2.putText(frame_for_drawing_np, label, (x_scaled, y_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 遍历 ignore 区域 (frame_data1)
    for _, row in frame_data1.iterrows():
        # 提取边界框信息 (原始坐标)
        bbox_left_orig = int(row['bbox_left'])
        bbox_top_orig = int(row['bbox_top'])
        bbox_width_orig = int(row['bbox_width'])
        bbox_height_orig = int(row['bbox_height'])

        # 缩放 ignore 区域坐标到新的尺寸
        x1_scaled = int(bbox_left_orig * scale_factor_w_final)
        y1_scaled = int(bbox_top_orig * scale_factor_h_final)
        x2_scaled = int((bbox_left_orig + bbox_width_orig) * scale_factor_w_final)
        y2_scaled = int((bbox_top_orig + bbox_height_orig) * scale_factor_h_final)

        # 确保坐标在图像范围内
        x1_scaled = max(0, min(new_w - 1, x1_scaled))
        y1_scaled = max(0, min(new_h - 1, y1_scaled))
        x2_scaled = max(0, min(new_w - 1, x2_scaled))
        y2_scaled = max(0, min(new_h - 1, y2_scaled))

        # 绘制填充的矩形在缩放后的图像上
        # 使用 frame_for_drawing_np
        cv2.rectangle(frame_for_drawing_np, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 0, 0), -1)  # 黑色填充，thickness 设置为 -1

    # 将修改后的帧写入输出视频
    # 使用 frame_for_drawing_np
    out.write(frame_for_drawing_np)

# 释放视频写入对象
out.release()
cv2.destroyAllWindows()
