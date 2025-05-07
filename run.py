import os
import argparse

import torch
from torch import optim
os.environ['TORCH_HOME'] = './.torch'

from utils.create import create_model, create_data
from utils.config import load_config



def run(config_path):
    # 1.导入设置
    cfg = load_config(config_path)
    # os.makedirs(f"experiments/{cfg['experiment_name']}/results", exist_ok=True)

    # 2.数据集创建
    train_loader, val_loader, test_loader, train_clean_loader, val_clean_loader = create_data(cfg)

    # 3.模型创建
    model = create_model(cfg)
    # 4.模型训练
    model.train_model(train_loader=train_loader, val_loader=val_loader, train_clean_loader=train_clean_loader,
                      val_clean_loader = val_clean_loader, num_epochs=cfg['train']['epochs'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/IA_YOLOV11.yaml",
        help="配置文件路径（默认: configs/exp1.yaml）"
    )
    args = parser.parse_args()
    run(args.config)
