import os
import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from utils.DataLoader import UAVDataLoaderBuilder
import importlib
from utils.config import load_config
from torchvision import transforms


def run_experiment(config_path):
    cfg = load_config(config_path)
    # logger = setup_logger(cfg['experiment_name'])

    # 创建结果目录
    os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

    # 1. 加载数据
    # path = cfg['dataset']['path'] / Path(f"{cfg['haze_method']}_{cfg['dataset']['data_path']}")
    #
    # if not os.path.exists(path):
    #     # 如果路径不存在，执行你想要的操作
    #     print(f"路径 {path} 不存在，正在执行相关操作...")
    #     # 导入模块
    #     module = importlib.import_module(f'haze.{cfg["haze_method"]}')
    #
    #     # 获取函数
    #     func = getattr(module, cfg['haze_method'])
    #
    #     # 调用函数并传入参数
    #     func('参数A', '参数B')
    #
    #     # 在此添加你的操作，例如创建文件夹、下载数据等

    builder = UAVDataLoaderBuilder(cfg)

    # 创建一个带固定种子的 Generator
    generator = torch.Generator()
    generator.manual_seed(cfg['seed'])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset, val_dataset, test_dataset, clean_dataset = builder.build(train_ratio=0.7, val_ratio=0.2, transform=transform)

    from PIL import Image
    for i in range(len(train_dataset)):
        img_path = train_dataset.image_files[i]
        label = train_dataset.label_files [i]
        img = Image.open(img_path).convert('RGB')
        if img.size != (1024, 540):  # 或你希望的尺寸
            print(f"[BAD SIZE] Index: {i}, Label: {label}, Size: {img.size}, Path: {img_path}")

    # 构造共享的 sampler（假设 train_dataset 和 clean_dataset 是一一对应的）
    sampler = RandomSampler(train_dataset, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    if cfg['is_clean']:
        clean_loader = DataLoader(clean_dataset, batch_size=8, sampler=sampler)
    for i, ((_, train_labels), (_, clean_labels)) in enumerate(zip(train_loader, clean_loader)):
        print(f"Batch {i + 1} Labels:")
        print("Train labels:", train_labels)
        print("Clean labels:", clean_labels)

        equal = torch.equal(train_labels, clean_labels)
        print("Labels are equal:", equal)
        print("-" * 30)

        if i == 2:  # 只比较前3个batch
            break

    # # 3. 加载检测器和跟踪器
    # detector = load_detector(cfg['detector'])
    # tracker = load_tracker(cfg['tracker'])
    #
    # all_detections, all_tracks = [], []
    #
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
