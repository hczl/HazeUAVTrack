import math
import os
import argparse
import cv2

import numpy as np
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as F
# Assuming these imports are correct based on your code structure
from utils.DataLoader import UAVDataLoaderBuilder
from utils.common import create_model, create_data
from utils.config import load_config

# 1.导入设置
cfg = load_config('configs/exp1.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# 3.模型创建
model = create_model(cfg)

# Load model weights
# model.train_model(train_loader=train_loader, val_loader=val_loader, clean_loader=clean_loader, num_epochs=cfg['train']['epochs'])
# model.load_state_dict(torch.load('models/IA_YOLOV3/checkpoints/checkpoint_epoch_80.pth'))
model.load_checkpoint('models/IA_YOLOV3/checkpoints/best_model.pth')


# Load the image
image_path = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M0101/img000002.jpg'
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor()
])
image = transform(image)
# 检查图片是否成功加载
if image is None:
    print(f"错误：无法加载图片。请检查路径是否正确: {image_path}")
    exit()

# --- 图像缩放处理 ---
# Get original image dimensions
original_h, original_w = image.shape[1], image.shape[2]  # Get original height and width

# Define a max size, e.g., 640
max_size = 640

# Calculate ratio for resizing while maintaining aspect ratio
r = min(1.0, max_size / original_w)
r = min(r, max_size / original_h)

# Calculate new dimensions based on the ratio
new_h = int(round(original_h * r))
new_w = int(round(original_w * r))

# Ensure new dimensions are multiples of 32
new_h = max(32, math.floor(new_h / 32) * 32)  # Ensure minimum size is 32x32
new_w = max(32, math.floor(new_w / 32) * 32)

# Recalculate scale factors based on the *final* new_h, new_w
scale_factor_w_final = new_w / original_w if original_w > 0 else 1.0
scale_factor_h_final = new_h / original_h if original_h > 0 else 1.0

# Resize the image
# Note: transforms.functional.resize expects (H, W) tuple for size
# Ensure image is float before resizing if necessary, or handle type appropriately
# Assuming input image is already a tensor suitable for resizing (e.g., float or uint8)

img_resized = F.resize(image, (new_h, new_w))

# Convert the resized tensor to NumPy format
img_resized_np = img_resized.permute(1, 2, 0).byte().numpy()

# Show image with OpenCV
# cv2.imshow("Annotated Image", img_resized_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert numpy array to torch tensor, change shape (H, W, C) to (C, H, W), add batch dimension
input_tensor = torch.from_numpy(img_resized_np).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
print(input_tensor)
# Model Inference
with torch.no_grad():
    # Pass the tensor to the predict method
    # Note: Some predict methods can also take the original image path/array directly
    # If your model's predict method expects the original image for plotting, you might need to adjust
    # However, the ultralytics Results object usually handles this if the input is a tensor.
    # results = model.predict(input_tensor)

    result = model(input_tensor, detach_dip = True)
img_np = result[0].permute(1, 2, 0).cpu().numpy()
if img_np.dtype != 'uint8':
    img_np = (img_np * 255).clip(0, 255).astype('uint8')
img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# 5. 用 OpenCV 显示
cv2.imshow("Image", img_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Assume results is a list containing one Results object for the single image
# if results:
#     # Get the first (and likely only) Results object
#     result = results[0]
#
#     # Print box information (optional, for verification)
#     print("Boxes (xywh):", result.boxes.xywh.shape)
#     # print("Boxes (xywhn):", result.boxes.xywhn)
#     # print("Boxes (xyxy):", result.boxes.xyxy)
#     # print("Boxes (xyxyn):", result.boxes.xyxyn)
#     # print("Confidence:", result.boxes.conf)
#     # print("Class IDs:", result.boxes.cls)
#
#     # --- Get the annotated image ---
#     # The plot() method draws the boxes, labels, etc., onto the image used for inference
#     # It returns a NumPy array suitable for OpenCV (BGR format, uint8)
#     annotated_image = result.plot()
#
#     # --- Display the annotated image ---
#     cv2.imshow("Annotated Image", annotated_image)
#
#     # Wait indefinitely until a key is pressed
#     cv2.waitKey(0)
#
#     # Close all OpenCV windows
#     # cv2.destroyAllWindows()
#
#     # # --- Save the annotated image (Optional) ---
#     # output_filename = "annotated_image_with_detections.jpg"
#     # cv2.imwrite(output_filename, annotated_image)
#     # print(f"Saved annotated image to {output_filename}")
#
#     # --- Alternative: Use the built-in save method ---
#     # This saves the annotated image to the default run directory (e.g., runs/detect/predictX)
#     # result.save()
#     # print(f"Saved annotated image using result.save() to {result.save_dir}")
#
# else:
#     print("No results obtained from prediction.")
#
# print("Script finished.")
