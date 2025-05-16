import sys

import importlib


import cv2
import numpy as np

def call_function(method_name, module_prefix, *args):
    module_name = f'{module_prefix}.{method_name}'
    spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    func = getattr(module, method_name)
    return func(*args)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms.functional as TF

def visualize_predictions(image_tensor, decoded_preds, targets, conf_thresh=0.05):
    """
    Args:
        image_tensor: torch.Tensor [3, H, W], 原图像张量（0-1 范围）
        decoded_preds: Tensor [N, 5], 每行为 [x1, y1, x2, y2, conf]
        targets: List, 每项为 [cls_id, _, x1, y1, w, h] 的格式
    """
    # 准备图像
    image_np = TF.to_pil_image(image_tensor.cpu()).copy()
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    # 绘制预测框
    for box in decoded_preds:
        x1, y1, x2, y2, conf = box.tolist()
        if conf < conf_thresh:
            continue
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'conf={conf:.2f}', color='lime', fontsize=10, weight='bold')

    # 绘制 GT 框
    for ann in targets:
        _, _, x, y, w, h = ann  # [cls_id, _, x, y, w, h]
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y - 5, 'GT', color='red', fontsize=10)

    ax.set_title('Predictions (green) vs Ground Truth (red)', fontsize=14)
    plt.axis('off')
    plt.show()

