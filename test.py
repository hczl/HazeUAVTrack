import os
import argparse

import torch
from torchvision.utils import save_image

os.environ['TORCH_HOME'] = './.torch'

from utils.create import create_model, create_data
from utils.config import load_config


def run(config_path):
    # 1.导入设置
    cfg = load_config(config_path)

    # 2.数据集创建
    _, _, test_loader, _, _ = create_data(cfg)

    # 3.模型创建
    model = create_model(cfg)
    model.load_model()
    model.eval()  # 设置为评估模式

    # 4.对test_loader中的第一张图片进行预测
    with torch.no_grad():
        for batch in test_loader:
            inputs , _, _ = batch
            inputs = inputs.to(cfg['device'])
            outputs = model.predict(inputs)

            # 保存预测结果或打印输出
            save_image(outputs[0], "predicted_result.png")  # 假设是图像
            print("已保存第一张图像的预测结果为 predicted_result.png")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/.yaml",
        help="配置文件路径（默认: configs/exp1.yaml）"
    )
    args = parser.parse_args()
    run(args.config)
