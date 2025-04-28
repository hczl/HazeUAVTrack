import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import ultralytics
from torch import optim
from tqdm import tqdm
from ultralytics import YOLO
import yaml

from .yolov3 import convert_targets_to_yolo, process_and_return_loaders, changeed__call__
from .IA_config import Settings

from .CNN_PP import CNN_PP
from .DIP import DIP_Module
from models.IA_YOLOV3.yolo_utils import AtmLight,DarkIcA,DarkChannel

from .yolo_utils import load_config
import torch.nn.functional as F

class IA_YOLOV3(nn.Module):
    def __init__(self, config):

        super(IA_YOLOV3, self).__init__()
        self.val_batch_nums = None
        self.train_batch_nums = None
        self.IA_config = Settings()
        self.config = config
        self.cnn_pp = CNN_PP(self.IA_config.dip_nums)
        self.dip_module = DIP_Module(self.IA_config)
        # 从pytorch hub下载yolov3
        print("Loading YOLOv3u model using ultralytics.YOLO...")
        # 使用 YOLO("yolov3u.pt") 加载模型。
        self.yolov3_wrapper = YOLO('yolov3.yaml').load('yolov3u.pt')
        self.yolov3 = self.yolov3_wrapper.model  # 用 nn.Module 直接 forward
        ultralytics.utils.loss.v8DetectionLoss.__call__ = changeed__call__
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放 GPU 内存（如果使用了）
        # self.yolov3.info()

    def forward(self, inputs, detach_dip=False, yolo_forward= False):
        n, c, _, _ = inputs.shape
        resized_inputs = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
        dark = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]),
                           dtype=torch.float32, device=self.config['device'])
        defog_A = torch.zeros((inputs.shape[0], inputs.shape[1]), dtype=torch.float32, device=self.config['device'])
        IcA = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.float32,
                          device=self.config['device'])

        for i in range(inputs.shape[0]):
            train_data_i = inputs[i]
            dark_i = DarkChannel(train_data_i)
            defog_A_i = AtmLight(train_data_i, dark_i)
            IcA_i = DarkIcA(train_data_i, defog_A_i)
            dark[i, ...] = dark_i
            defog_A[i, ...] = defog_A_i
            IcA[i, ...] = IcA_i
        IcA = torch.unsqueeze(IcA, dim=1)

        filter_features = self.cnn_pp(resized_inputs)
        # print("filter_features", filter_features.shape)
        dip_output = self.dip_module(inputs, filter_features, defog_A, IcA)

        if detach_dip:
            return dip_output
        elif self.training and yolo_forward:
            # print("dip", dip_output.shape)
            yolov3_output = self.yolov3(dip_output)
            return yolov3_output
        else:
            yolov3_output = self.yolov3(dip_output)
            return yolov3_output[0]  # 推理时返回预测框

        # if self.training:
        #     loss_outputs = self.yolov3.loss(yolov3_outputs[1], targets)  # 使用多尺度 raw output 计算 loss
        #     return loss_outputs  # 是一个 dict: {box, cls, dfl, loss}
        # else:
        #     return yolov3_outputs[0]  # 只返回最终预测框用于推理
        # yolov3_predictions = self.yolov3_model(dip_processed_output) # DIP output to YOLOv3
        # return cnn_pp_output, yolov3_predictions # Return both outputs for loss calculation if needed

    def train_step(self, tra_batch, clean_batch):
        torch.autograd.set_detect_anomaly(True)
        targets_dip, _ = clean_batch
        low_res_images, targets_yolov3 = tra_batch
        # print(targets_yolov3)
        # print(type(targets_yolov3))
        low_res_images = low_res_images.to(self.device)
        targets_dip = targets_dip.to(self.device)

        self.optimizer.zero_grad()
        # print(targets_yolov3[0].shape)
        dip_output= self(low_res_images, detach_dip=True)

        dip_loss = self.calculate_dip_loss(dip_output, targets_dip)
        dip_loss.backward(retain_graph=True)
        # forward 返回 DIP 输出 和 yolov3 的 multi-scale 输出
        yolov3_output = self(low_res_images,yolo_forward=True)
        yolov3_loss_tuple = self.yolov3_wrapper.loss(targets_yolov3, yolov3_output)
        yolov3_loss = sum([t.sum() for t in yolov3_loss_tuple])

        yolov3_loss.backward()

        self.optimizer.step()

        return {
            'yolov3_loss': yolov3_loss.item(),
            'dip_loss': dip_loss.item()
        }

    def train_epoch(self, train_loader, clean_loader, epoch):
        epoch_yolov3_loss = 0.0
        epoch_dip_loss = 0.0
        train_loader, _ = process_and_return_loaders(train_loader)
        clean_loader, _ = process_and_return_loaders(clean_loader)
        pbar = tqdm(zip(train_loader, clean_loader), total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, (tra_batch, clean_batch) in enumerate(pbar):
            train_info = self.train_step(tra_batch, clean_batch)
            yolov3_loss = train_info['yolov3_loss']
            dip_loss = train_info['dip_loss']

            epoch_yolov3_loss += yolov3_loss
            epoch_dip_loss += dip_loss

            if batch_idx % self.config['train']['log_interval'] == 0:
                pbar.set_postfix({
                    'YOLOv3 Loss': f'{yolov3_loss:.4f}',
                    'DIP Loss': f'{dip_loss:.4f}',
                    'Batch': f'{batch_idx + 1}/{self.train_batch_nums}'
                })

        avg_yolov3_loss = epoch_yolov3_loss / self.train_batch_nums
        avg_dip_loss = epoch_dip_loss / self.train_batch_nums

        print(f"Epoch {epoch} 训练完成，平均 YOLOv3 Loss: {avg_yolov3_loss:.4f}, 平均 DIP Loss: {avg_dip_loss:.4f}")
        return {'avg_yolov3_loss': avg_yolov3_loss, 'avg_dip_loss': avg_dip_loss}

    def evaluate(self, val_loader):
        total_yolov3_loss = 0.0
        val_loader, _ = process_and_return_loaders(val_loader)
        with torch.no_grad():
            pbar = tqdm(val_loader, total=self.val_batch_nums, desc=f"Val")
            for batch_idx, val_batch in enumerate(pbar):
                # 获取输入和标签
                low_res_images, targets_yolov3 = val_batch

                # 移动到设备
                low_res_images = low_res_images.to(self.device)

                # YOLO前向
                yolov3_output = self(low_res_images, yolo_forward=True)
                yolov3_loss_tuple = self.yolov3_wrapper.loss(targets_yolov3, yolov3_output)
                yolov3_loss = sum([t.sum() for t in yolov3_loss_tuple])

                total_yolov3_loss += yolov3_loss.item()
                if batch_idx % self.config['train']['log_interval'] == 0:
                    pbar.set_postfix({
                        'YOLOv3 Loss': f'{yolov3_loss:.4f}',
                        'Batch': f'{batch_idx + 1}/{self.val_batch_nums}'
                    })

        avg_yolov3_loss = total_yolov3_loss / self.val_batch_nums

        print(
            f"验证集 Avg YOLOv3 Loss: {avg_yolov3_loss:.4f}")
        return {
            'avg_yolov3_loss': avg_yolov3_loss
        }

    def predict(self, high_res_images):
        self.yolov3.eval()
        self.cnn_pp.eval()
        self.dip_module.eval()
        high_res_images = high_res_images.to(self.device)
        with torch.no_grad():
            dip_processed_output = self(high_res_images, detach_dip=True)
            results = self.yolov3_wrapper.predict(dip_processed_output)
        return results

    def calculate_dip_loss(self, dip_output, targets_dip):
        criterion = nn.MSELoss()
        return criterion(dip_output, targets_dip)


    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        self.config = checkpoint['config']
        print(f"检查点已加载，从 epoch {start_epoch} 继续训练 (如果需要)")
        return start_epoch

    def unfreeze_yolov3_backbone(self):
        print("==> 解冻 YOLOv3 全部层（开始 Fine-tuning）")
        for param in self.yolov3.parameters():
            param.requires_grad = True

        # 重新构建 optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['train']['lr']
        )

    def train_model(self, train_loader, val_loader, clean_loader, start_epoch=0, num_epochs=100, checkpoint_dir='./models/IA_YOLOV3/checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_loss = float('inf')
        # 训练前预处理
        _, self.train_batch_nums= process_and_return_loaders(train_loader)
        _, self.val_batch_nums = process_and_return_loaders(val_loader)
        if self.config['train']['resume_training']:
            print("==> 尝试加载最近 checkpoint ...")
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path)
            else:
                print("未找到 checkpoint，重新训练。")

        for epoch in range(start_epoch, num_epochs):
            # 冻结训练阶段（预热）
            if epoch < self.IA_config.warmup_epochs:
                for i, layer in enumerate(self.yolov3.model):
                    requires_grad = False if i <= 10 else True
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
            # 解冻阶段（fine-tune）
            elif epoch == self.IA_config.warmup_epochs:
                self.unfreeze_yolov3_backbone()

            self.yolov3.train()
            self.cnn_pp.train()
            self.dip_module.train()
            # print(f"\nEpoch {epoch + 1}/{num_epochs}")
            # # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

            self.train_epoch(train_loader, clean_loader, epoch)

            # 验证集
            if val_loader:
                self.yolov3.eval()
                self.cnn_pp.eval()
                self.dip_module.eval()

                val_stats = self.evaluate(val_loader)

                # 保存最优模型
                if val_stats['avg_yolov3_loss'] < best_loss:
                    best_loss = val_stats['avg_yolov3_loss']
                    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    self.save_checkpoint(epoch, best_path)
                    print(f"==> Best model saved to {best_path}")

            # 定期保存 checkpoint
            if (epoch + 1) % self.config['train']['checkpoint_save_interval'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch, checkpoint_path)
                print(f"==> Checkpoint saved to {checkpoint_path}")

        print("训练完成！")


if __name__ == '__main__':
    import sys

    from torch.utils.data import DataLoader, RandomSampler
    current_file_path = os.path.abspath(__file__)
    # print(sys.path)

    current_dir = os.path.dirname(current_file_path)

    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # 4. 将项目根目录添加到 sys.path 的最前面
    # 这样做是为了确保 Python 优先在你的项目根目录中查找模块
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from torchvision import transforms

    cfg = load_config(r'D:\FFFFFiles\bis_sheji\configs\exp1.yaml')
    from utils.DataLoader import UAVDataLoaderBuilder, custom_collate_fn
    builder = UAVDataLoaderBuilder(cfg)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset, val_dataset, test_dataset, clean_dataset = builder.build(train_ratio=0.7, val_ratio=0.2, transform=transform)
    generator_train = torch.Generator().manual_seed(cfg['seed'])
    sampler_train = RandomSampler(train_dataset, generator=generator_train)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler_train, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # 3. Instantiate the model
    model = IA_YOLOV3(cfg)

    # 4. Move model to device (if you have CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device # Add device attribute for model to use

    # 5. Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.optimizer = optimizer # Add optimizer to the model

    # 6. Test forward pass
    dummy_input = torch.randn(4, 3, 1024, 540).to(device) # Batch of 2, 3 channels, 512x512
    # output = model(dummy_input,dummy_input)
    def get_shape(a):
        shape = []
        try:
            while True:
                shape.append(len(a))
                a = a[0]
        except:
            pass
        return shape
    # print(len(output))
    # model.train_model(dummy_input,)
    # print(get_shape(output[1]))
    # cnn_pp_output, yolov3_predictions = model(dummy_input)
    # print("CNN_PP Output Shape:", cnn_pp_output[0].shape) # Print shape of filter_features
    # print("YOLOv3 Predictions Shape:", yolov3_predictions.shape)
    #
    # # 7. Test training for a few epochs
    # print("\n--- Starting Training Test ---")
    # model.train_model(train_dataloader, val_dataloader, num_epochs=2, checkpoint_dir='./test_checkpoints') # Train for 2 epochs
    #
    # # 8. Test checkpoint saving and loading
    # checkpoint_path = './test_checkpoints/checkpoint_epoch_1.pth' # Path to the saved checkpoint from epoch 1
    # loaded_start_epoch = model.load_checkpoint(checkpoint_path)
    # print(f"\nLoaded checkpoint, starting epoch: {loaded_start_epoch}")
    #
    # print("\n--- Testing Completed ---")
