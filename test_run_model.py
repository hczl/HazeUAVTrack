import os
import argparse
import cv2

import numpy as np
import torch
from torch import optim

# Assuming these imports are correct based on your code structure
from utils.DataLoader import UAVDataLoaderBuilder
from utils.common import create_model, create_data
from utils.config import load_config

# 1.导入设置
cfg = load_config('configs/exp2.yaml')
os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

# 3.模型创建
model = create_model(cfg)

# Load model weights
# model.train_model(train_loader=train_loader, val_loader=val_loader, clean_loader=clean_loader, num_epochs=cfg['train']['epochs'])
# model.load_state_dict(torch.load('models/IA_YOLOV3/checkpoints/checkpoint_epoch_80.pth'))
model.load_checkpoint('models/DE_NET/checkpoints/best_model.pth')

# Set models to evaluation mode
model.enhancement.eval()
model.yolov3.eval()

# Load the image
image_path = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M0101/img000002.jpg'
image = cv2.imread(image_path)

# 检查图片是否成功加载
if image is None:
    print(f"错误：无法加载图片。请检查路径是否正确: {image_path}")
    exit()

# --- 图像缩放处理 ---
# Get original image dimensions
h_orig, w_orig = image.shape[:2]

target_size = 1024

# Calculate scale based on the longer side
scale = target_size / max(h_orig, w_orig)

# Calculate intermediate dimensions after scaling
h_inter = int(h_orig * scale)
w_inter = int(w_orig * scale)

# Pad to a multiple of 32 (common for CNNs)
# Calculate padding needed
pad_w = (32 - (w_inter % 32)) % 32
pad_h = (32 - (h_inter % 32)) % 32

# New dimensions after padding
w_new = w_inter + pad_w
h_new = h_inter + pad_h

# Resize the image
# Note: OpenCV resize expects (width, height)
resized_image = cv2.resize(image, (w_inter, h_inter), interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR or INTER_AREA for downsampling

# Pad the image
# Note: OpenCV pad expects (top, bottom, left, right)
padded_image = cv2.copyMakeBorder(resized_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # Pad with gray color

# Store the original image for later plotting (optional, but good practice)
# annotated_image = image.copy() # If you want to draw on the original size

# --- 缩放和填充处理结束 ---


# Convert BGR to RGB (for model input)
input_image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

# Convert to float and normalize to [0, 1]
input_tensor_np = input_image_rgb.astype(np.float32) / 255.0

# Convert numpy array to torch tensor, change shape (H, W, C) to (C, H, W), add batch dimension
input_tensor = torch.from_numpy(input_tensor_np).permute(2, 0, 1).unsqueeze(0).float().to('cuda')

# Model Inference
with torch.no_grad():
    # Pass the tensor to the predict method
    # Note: Some predict methods can also take the original image path/array directly
    # If your model's predict method expects the original image for plotting, you might need to adjust
    # However, the ultralytics Results object usually handles this if the input is a tensor.
    results = model.predict(input_tensor)

# Assume results is a list containing one Results object for the single image
if results:
    # Get the first (and likely only) Results object
    result = results[0]

    # Print box information (optional, for verification)
    print("Boxes (xywh):", result.boxes.xywh.shape)
    # print("Boxes (xywhn):", result.boxes.xywhn)
    # print("Boxes (xyxy):", result.boxes.xyxy)
    # print("Boxes (xyxyn):", result.boxes.xyxyn)
    # print("Confidence:", result.boxes.conf)
    # print("Class IDs:", result.boxes.cls)

    # --- Get the annotated image ---
    # The plot() method draws the boxes, labels, etc., onto the image used for inference
    # It returns a NumPy array suitable for OpenCV (BGR format, uint8)
    # annotated_image = result.plot()

    # --- Display the annotated image ---
    # cv2.imshow("Annotated Image", annotated_image)

    # Wait indefinitely until a key is pressed
    # cv2.waitKey(0)

    # Close all OpenCV windows
    # cv2.destroyAllWindows()

    # # --- Save the annotated image (Optional) ---
    # output_filename = "annotated_image_with_detections.jpg"
    # cv2.imwrite(output_filename, annotated_image)
    # print(f"Saved annotated image to {output_filename}")

    # --- Alternative: Use the built-in save method ---
    # This saves the annotated image to the default run directory (e.g., runs/detect/predictX)
    # result.save()
    # print(f"Saved annotated image using result.save() to {result.save_dir}")

else:
    print("No results obtained from prediction.")

print("Script finished.")
