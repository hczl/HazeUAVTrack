import cv2
import pandas as pd
import os

# 读取标签数据
data = pd.read_csv('data/UAV-M/UAV-benchmark-MOTD_v1.0/GT/M0201_gt_whole.txt', header=None,
                   names=['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
                          'out_of_view', 'occlusion', 'object_category'])

# 图像文件夹路径
image_folder = 'data/MiDaS_Deep_UAV-benchmark-M/M0201'

# 获取所有图像文件名并排序，确保按照文件名中的数字部分排序
image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split('img')[1].split('.')[0]))

# 输出视频的设置
frame_width = 1024  # 设置输出视频的宽度
frame_height = 540  # 设置输出视频的高度
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

# 遍历每一帧图片
for image_file in image_files:
    # 获取当前帧图像路径
    img_path = os.path.join(image_folder, image_file)

    # 读取图像帧
    frame = cv2.imread(img_path)

    # 获取当前帧的索引
    frame_index = int(image_file.split('img')[1].split('.')[0])  # 提取文件名中的数字部分

    # 获取当前帧的标签数据
    frame_data = data[data['frame_index'] == frame_index]

    # 遍历该帧的所有目标
    for _, row in frame_data.iterrows():
        # 提取边界框信息
        x, y, w, h = int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])

        # 获取目标ID作为标签
        label = f"ID: {row['target_id']}"

        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 使用绿色框

        # 在左上角添加标签
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将修改后的帧写入输出视频
    out.write(frame)

# 释放视频写入对象
out.release()
cv2.destroyAllWindows()
