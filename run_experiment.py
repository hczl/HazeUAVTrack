import os
import yaml
import cv2
import argparse

from torch.utils.data import DataLoader

from utils.DataLoader import UAVDataLoaderBuilder
from dehaze import apply_dehaze
from detectors import load_detector
from trackers import load_tracker
from evaluation import evaluate_detection, evaluate_tracking
from visualizer import draw_detections, draw_tracks
from utils.config import load_config
from utils.logger import setup_logger


def run_experiment(config_path):
    cfg = load_config(config_path)
    logger = setup_logger(cfg['experiment_name'])

    # 创建结果目录
    os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

    # 1. 加载数据
    builder = UAVDataLoaderBuilder(cfg)
    train_dataset, val_dataset, test_dataset = builder.build(train_ratio=0.6, val_ratio=0.2)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


    # 3. 加载检测器和跟踪器
    detector = load_detector(cfg['detector'])
    tracker = load_tracker(cfg['tracker'])

    all_detections, all_tracks = [], []

    for img_path in dataset:
        image = cv2.imread(img_path)
        if dehaze_method:
            image = apply_dehaze(image, method=dehaze_method)

        detections = detector.predict(image)
        tracks = tracker.update(detections, image)

        all_detections.append(detections)
        all_tracks.append(tracks)

        # 可视化
        vis = draw_detections(image.copy(), detections)
        vis = draw_tracks(vis, tracks)
        cv2.imwrite(f"experiments/{cfg['experiment_name']}/results/{os.path.basename(img_path)}", vis)

    # 评估
    evaluate_detection(all_detections)
    evaluate_tracking(all_tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp1.yaml",
        help="配置文件路径（默认: configs/exp1.yaml）"
    )
    args = parser.parse_args()
    run_experiment(args.config)
