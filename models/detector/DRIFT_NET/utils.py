import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F_tv
from PIL import Image

def visualize_predictions(image_tensor, predictions, targets, ignore_list=None, conf_threshold=0.5, title="Predictions & Targets"):
    """
    Visualizes predictions and ground truth boxes on an image.

    Args:
        image_tensor (Tensor): Input image tensor [C, H, W] in [0, 1] or [0, 255] range.
        predictions (Tensor): Predicted boxes [N, 5] in [x1, y1, x2, y2, conf] format.
        targets (list): List of GT annotations [class_id, track_id, x, y, w, h].
        ignore_list (list, optional): List of ignore annotations [class_id, track_id, x, y, w, h].
        conf_threshold (float): Confidence threshold for displaying predictions.
        title (str): Plot title.
    """
    # Ensure image is in [0, 255] uint8 format for drawing
    if image_tensor.max() <= 1.0:
        image_tensor = (image_tensor * 255).byte()
    else:
        image_tensor = image_tensor.byte()

    img_H, img_W = image_tensor.shape[1:]

    # Filter predictions by confidence
    if predictions.numel() > 0:
        keep = predictions[:, 4] > conf_threshold
        predictions = predictions[keep]

    # Prepare boxes and labels for drawing
    all_boxes = []
    all_colors = []
    all_labels = []

    # Add predictions
    if predictions.numel() > 0:
        pred_boxes = predictions[:, :4]
        pred_scores = predictions[:, 4]
        all_boxes.append(pred_boxes)
        all_colors.extend(['red'] * pred_boxes.shape[0])
        all_labels.extend([f'{s:.2f}' for s in pred_scores.tolist()]) # Display confidence

    # Add GT targets (convert xywh to xyxy)
    if targets:
        gt_boxes_xyxy = []
        gt_labels = []
        for ann in targets:
            x, y, w, h = float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5])
            gt_boxes_xyxy.append([x, y, x + w, y + h])
            gt_labels.append(f'GT Class {ann[0]} Track {ann[1]}') # Display class and track ID
        if gt_boxes_xyxy:
            all_boxes.append(torch.tensor(gt_boxes_xyxy, device=image_tensor.device))
            all_colors.extend(['green'] * len(gt_boxes_xyxy))
            all_labels.extend(gt_labels)

    # Add Ignore regions (convert xywh to xyxy)
    if ignore_list:
        ignore_boxes_xyxy = []
        ignore_labels = []
        for ann in ignore_list:
            x, y, w, h = float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5])
            ignore_boxes_xyxy.append([x, y, x + w, y + h])
            ignore_labels.append(f'Ignore Class {ann[0]} Track {ann[1]}') # Display class and track ID
        if ignore_boxes_xyxy:
            all_boxes.append(torch.tensor(ignore_boxes_xyxy, device=image_tensor.device))
            all_colors.extend(['yellow'] * len(ignore_boxes_xyxy))
            all_labels.extend(ignore_labels)


    if all_boxes:
        all_boxes_tensor = torch.cat(all_boxes, dim=0)
        # Ensure boxes are within image bounds for drawing
        all_boxes_tensor[:, [0, 2]] = all_boxes_tensor[:, [0, 2]].clamp(0, img_W)
        all_boxes_tensor[:, [1, 3]] = all_boxes_tensor[:, [1, 3]].clamp(0, img_H)

        # Draw boxes
        output_image = draw_bounding_boxes(
            image_tensor,
            all_boxes_tensor,
            colors=all_colors,
            labels=all_labels,
            width=2
        )
    else:
        output_image = image_tensor # No boxes to draw

    # Convert back to PIL Image for display
    output_image_pil = F_tv.to_pil_image(output_image)

    # Display the image
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image_pil)
    plt.title(title)
    plt.axis('off')
    plt.show()


