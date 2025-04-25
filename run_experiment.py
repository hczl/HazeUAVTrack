import os
import argparse

import torch
from torch import optim


from utils.common import create_model, create_data
from utils.config import load_config



def run_experiment(config_path):
    # 1.导入设置
    cfg = load_config(config_path)
    os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

    # 2.数据集创建
    train_loader, val_loader, test_loader, clean_loader = create_data(cfg)

    # 3.模型创建
    model = create_model(cfg)
    # 4.模型训练
    model.train_model(train_loader=train_loader, val_loader=val_loader, clean_loader=clean_loader, num_epochs=cfg['train']['epochs'])
    # for img_path in dataset:
    #     image = cv2.imread(img_path)
    #     if dehaze_method:
    #         image = apply_dehaze(image, method=dehaze_method)
    #
    #     detections = detector.predict(image)
    #     tracks = tracker.update(detections, image)
    #
    #     all_detections.append(detections)
    #     all_tracks.append(tracks)
    #
    #     # 可视化
    #     vis = draw_detections(image.copy(), detections)
    #     vis = draw_tracks(vis, tracks)
    #     cv2.imwrite(f"experiments/{cfg['experiment_name']}/results/{os.path.basename(img_path)}", vis)
    #
    # # 评估
    # evaluate_detection(all_detections)
    # evaluate_tracking(all_tracks)


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
