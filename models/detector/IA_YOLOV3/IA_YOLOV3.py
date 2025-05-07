import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression

from models.detector.DE_NET.utils import changeed__call__, process_batch
from .DIP import DIP


class IA_YOLOV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dip = DIP()

        # 用ultralytics载入模型并提取其 nn.Module（只做一次）
        yolov3_wrapper = YOLO('yolov3.yaml')
        yolov3_wrapper.load('models/detector/YOLOV3/yolov3u.pt')
        self.yolov3 = yolov3_wrapper.model
        self.loss_fn = yolov3_wrapper.loss

        # monkey patch loss
        v8DetectionLoss.__call__ = changeed__call__

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def forward(self, x): # x 是原始输入图像
        out = self.dip(x)
        out = self.yolov3(out)
        return out

    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.95, iou_thresh=0.45):
        self.eval()
        self.yolov3.eval()
        raw_output = self(high_res_images)
        return self.decode_output(raw_output, conf_thresh, iou_thresh)[0]

    def decode_output(self, raw_output, conf_thresh=0.95, iou_thresh=0.45, max_det=300):
        detections = non_max_suppression(
            raw_output,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            max_det=max_det,
            classes=None,
        )
        return detections

    def forward_loss(self, haze_imgs, targets, ignore_list):
        haze_imgs, targets, ignore_list = process_batch((haze_imgs, targets, ignore_list))
        yolov3_output = self(haze_imgs)
        yolov3_loss_tuple = self.loss_fn(targets, yolov3_output)
        all_loss_tensors = [loss for group in yolov3_loss_tuple for loss in group]
        yolov3_total_loss = sum(all_loss_tensors)
        return {
            'yolov3_loss': yolov3_total_loss,
            'total_loss': yolov3_total_loss,
        }