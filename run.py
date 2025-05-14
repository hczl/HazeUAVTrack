import os
import argparse

os.environ['TORCH_HOME'] = './.torch'

from utils.create import create_model, create_data
from utils.config import load_config

import torch # 需要导入 PyTorch 库
import gc    # 需要导入垃圾回收模块

def run_experiment(config_path):
    print(f"--- Starting experiment for {config_path} ---")

    # 1.导入设置
    cfg = load_config(config_path)

    # 2.数据集创建
    # 假设create_data不会长时间占用大量GPU内存
    train_loader, val_loader, test_loader, train_clean_loader, val_clean_loader = create_data(cfg)

    # 3.模型创建
    model = create_model(cfg)
    # 注意：create_model 或 train_model 内部会负责将模型移动到 GPU (model.to(device))

    # 4.模型训练
    print("Starting model training...")
    model.train_model(train_loader=train_loader, val_loader=val_loader, train_clean_loader=train_clean_loader,
                      val_clean_loader = val_clean_loader, num_epochs=cfg['train']['epochs'])
    print("Model training finished.")

    # --- 5. 内存清理步骤 ---
    print("Starting memory cleanup...")
    if torch.cuda.is_available():
        # 将模型移回 CPU
        # 确保模型在 GPU 上时才尝试移动，避免不必要的错误
        try:
            model.to('cpu')
            print("Model moved to CPU.")
        except Exception as e:
            print(f"Could not move model to CPU: {e}")
            # 如果模型不在GPU上，这个to('cpu')调用可能什么也不做，或者抛出错误，
            # 但通常在训练后模型都会在GPU上。

        # 删除模型变量的引用
        del model
        print("Model variable deleted.")

        # 清除 PyTorch 的 CUDA 内存缓存
        # 这是释放 GPU 显存的关键步骤
        torch.cuda.empty_cache()
        print("CUDA cache emptied.")

    # 强制运行 Python 垃圾回收
    # 有助于回收不再引用的 Python 对象，包括模型相关的其他结构
    gc.collect()
    print("Python garbage collection run.")

    print(f"--- Finished experiment for {config_path} ---")
    print("-" * 30) # 分隔线，让输出更清晰



if __name__ == "__main__":
    # 确保 PyTorch 能够访问到 GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Experiments will run on CPU (and likely be very slow). Memory issue less likely but cleanup is still good practice.")
    else:
         print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")


    yaml = ['AD_YOLOV11']
    for i in yaml:
        run_experiment(f"configs/{i}.yaml")

    print("\nAll experiments finished.")
    # 所有实验结束后，最后再清空一次缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # 所有实验完成后，最后清空一次 CUDA 缓存。
        print("Final CUDA cache cleared after all experiments.")
