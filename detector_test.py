import os
import time
import math
import torch
import numpy as np
import cv2 # 使用 OpenCV 进行视频写入和图像处理
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF # 别名，上面F已经导入了
from tqdm import tqdm

# Assuming these utility functions are in your project
from utils.config import load_config
from utils.create import create_model # This should create the FSDT model

# ---- 初始设置 ----
# os.environ['TORCH_HOME'] = './.torch' # Uncomment if you need to set TORCH_HOME
image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1005'  # 输入图像文件夹
output_folder = 'output/detection_results_video'  # 输出结果文件夹 (用于保存视频文件)
video_filename = 'output_video.mp4' # 输出视频文件名
yaml_path = 'configs/DE_NET.yaml'  # 你的配置 YAML 文件路径
max_size = 1024  # 图像resize的最大边长，与你的模型输入要求一致
conf_threshold = 0.25  # 检测置信度阈值
output_fps = 30 # 输出视频的帧率

# ---- 加载配置和模型 ----
print(f"加载配置: {yaml_path}")
cfg = load_config(yaml_path)

model = create_model(cfg)  # Assumes create_model returns an instance of FSDT
model.load_model()  # Load weights using the FSDT's load_model method
device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode
print(f"使用设备: {device}")

# ---- 图像处理和检测函数 ----
transform = transforms.Compose([transforms.ToTensor()])

def preprocess_image(image_pil, max_size=640):
    """Preprocesses PIL image: to tensor, resize, add batch dim, move to device."""
    orig_w, orig_h = image_pil.size
    image_tensor = transform(image_pil)

    # Calculate resize scale and new dimensions
    r = min(1.0, max_size / float(max(orig_w, orig_h)))
    new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
    new_w = max(32, int(math.floor(orig_w * r / 32) * 32))

    image_resized = F.resize(image_tensor, (new_h, new_w))
    input_tensor = image_resized.unsqueeze(0).to(device)

    # Return input tensor and original/new dimensions for scaling boxes
    return input_tensor, (orig_w, orig_h), (new_w, new_h)


def scale_boxes_to_original(boxes, orig_dims, new_dims):
    """Scales predicted boxes from resized image coords back to original image coords."""

    scaled_boxes = []
    # print(boxes)
    for box in boxes:
        # box format: [x1, y1, x2, y2, conf, ...]
        # Ensure box has at least 4 coordinates
        if len(box) < 4:
            continue
        scaled_boxes.append(box)
    return scaled_boxes


# ---- 获取图像列表 ----
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if not image_files:
    print(f"错误：未在文件夹 {image_folder} 中找到任何图像文件。")
    exit()

# ---- 创建输出文件夹 ----
os.makedirs(output_folder, exist_ok=True)
video_output_path = os.path.join(output_folder, video_filename)
print(f"视频将保存到: {video_output_path}")

# ---- OpenCV VideoWriter setup ----
out = None # VideoWriter object
showed_first_frame = False # Flag to show the first frame preview

# ---- 处理每张图像 ----
print("开始处理图像并写入视频...")
with torch.no_grad():
    for i, image_path in enumerate(tqdm(image_files, desc="处理图像")):
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"跳过文件 {image_path}: 无法打开或处理图像 ({e})")
            continue

        # 1. 预处理图像
        input_tensor, orig_dims, new_dims = preprocess_image(image_pil, max_size)

        # 2. 通过模型进行预测
        # model.predict first dehazes, then detects/tracks based on cfg flags
        # Since detector_flag=True and tracker_flag=False, it returns detector results
        # We need the dehazed image to draw on. Let's get it separately.
        dehazed_tensor = model.dehaze(input_tensor)

        # Convert dehazed tensor to OpenCV format (NumPy BGR uint8)
        # Shape (H, W, C), range [0, 255], type uint8
        # Permute from (C, H, W) to (H, W, C)
        dehazed_np = (dehazed_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV drawing functions
        dehazed_np = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)

        # Get predictions from the model (which uses the dehazed image internally)
        predictions = model.predict(input_tensor)
        # print(predictions)
        # 3. 缩放预测框到原始图像尺寸
        # predictions is a list of [x1, y1, x2, y2, conf, ...]
        scaled_predictions = scale_boxes_to_original(predictions, orig_dims, new_dims)

        # 4. 在图像上绘制结果
        img_to_draw = dehazed_np.copy() # Draw on a copy to be safe

        # Draw bounding boxes and labels
        # This loop correctly handles the case where scaled_predictions is empty
        for det in scaled_predictions:
            # Ensure the detection has at least 5 elements (x1, y1, x2, y2, conf)
            if len(det) < 5:
                continue # Skip malformed detections

            x1, y1, x2, y2, conf = det[:5] # Use only the first 5 elements

            # Ensure coordinates are integers for OpenCV drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw rectangle (OpenCV uses BGR color format)
            # Green color (0, 255, 0), thickness 2
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add text annotation (coordinates + confidence)
            # Position text slightly above the top-left corner
            text_x, text_y = x1, y1 - 10 # Offset slightly above the box
            # Ensure text_y is not negative (at least 15 pixels from top for visibility)
            text_y = max(text_y, 15)

            # Label format: "conf" or "x1,y1,x2,y2,conf"
            # Using just confidence is often less cluttered
            label = f"{conf:.2f}"
            # If you want coords too: label = f"{x1},{y1},{x2},{y2},{conf:.2f}"

            # Draw text (OpenCV uses BGR color format)
            # Yellow color (0, 255, 255), font scale 0.5, thickness 2
            cv2.putText(img_to_draw, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 5. 视频写入
        # Initialize VideoWriter object on the first frame
        if out is None:
            height, width, _ = img_to_draw.shape
            # Define the codec (e.g., 'mp4v', 'XVID', 'MJPG')
            # 'mp4v' is a common choice for MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            try:
                out = cv2.VideoWriter(video_output_path, fourcc, output_fps, (width, height))
                if not out.isOpened():
                     raise IOError("OpenCV VideoWriter object could not be opened.")
                print(f"成功创建视频写入对象，分辨率: {width}x{height}, 帧率: {output_fps}")
            except Exception as e:
                print(f"错误: 无法创建视频写入对象: {e}")
                print("后续帧将不会写入视频。请检查 OpenCV 安装和编码器兼容性。")
                out = None # Ensure out is None if creation failed
                # Decide if you want to stop here or continue processing without writing
                # For this script, let's continue but skip writing
                # break # Uncomment this line if you want to stop entirely

        # Show the first frame preview using OpenCV
        if not showed_first_frame and out is not None: # Only show if writer is valid
            cv2.imshow('First Frame with Detections (Press Any Key)', img_to_draw)
            print("显示第一帧预览，请按任意键继续...")
            cv2.waitKey(0) # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows() # Close the preview window
            showed_first_frame = True # Mark as shown

        # Write the processed frame to the video file
        if out is not None and out.isOpened():
             out.write(img_to_draw)
        # else:
        #     print(f"警告: 视频写入对象未打开或创建失败，跳过写入帧 {i}.") # Optional warning

# ---- 释放 VideoWriter 和清理 ----
if out is not None and out.isOpened():
    out.release()
    print(f"视频已保存到: {video_output_path}")
elif out is not None and not out.isOpened():
     print("视频写入对象创建失败或未成功打开，视频未保存。")
else: # out is None
     print("未找到图像文件或处理出错，视频写入对象未创建。")


cv2.destroyAllWindows() # Ensure all OpenCV windows are closed
print("处理完成。")

