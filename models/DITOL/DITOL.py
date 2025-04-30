import os

import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from .FALCON.FALCON import Falcon
from .FALCON.perceptual import PerceptualNet
from .FALCON.utils import make_dark_channel_tensor
from .utils import process_batch


class DITOL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.val_batch_nums = None
        self.train_batch_nums = None
        self.falcon = Falcon(config)

        # loss
        self.mse = nn.MSELoss()
        self.perc_loss_network = PerceptualNet(net=self.falcon.falcon_config.train['perceptual']['net'],
                                          style_layers=self.falcon.falcon_config.train['perceptual']['style'],
                                          content_layers=self.falcon.falcon_config.train['perceptual']['content'], device=self.config['device'])
    def forward(self, input):
        x = self.falcon(input)
        return x

    def train_step(self, tra_batch, clean_batch):
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)

        low_res_images, targets, _ = process_batch(tra_batch)
        clean_image, _, _ = process_batch(clean_batch)
        low_res_images = low_res_images.to(self.config['device'])
        clean_image = clean_image.to(self.config['device'])

        t_gt = make_dark_channel_tensor(clean_image)
        # --- Training steps ---
        dehazed, t_haze = self.forward(low_res_images)
        loss_img = self.mse(dehazed, clean_image)  # MSELoss
        loss_map = self.mse(t_haze.to(self.device), t_gt.to(self.device))
        loss_perc = self.perc_loss_network(dehazed, clean_image) if self.falcon.falcon_config.train['perceptual'][
                                                          'net'] is not False else torch.Tensor([0.]).to(self.device)
        loss_final = (self.falcon.falcon_config.train['loss_ratio'][0] * loss_img +
                      self.falcon.falcon_config.train['loss_ratio'][1] * loss_map +
                      self.falcon.falcon_config.train['loss_ratio'][2] * loss_perc)

        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()
        return loss_final

    def train_epoch(self, train_loader, clean_loader, epoch):
        epoch_loss = 0.0

        pbar = tqdm(zip(train_loader, clean_loader), total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, (tra_batch, clean_batch) in enumerate(pbar):
            loss = self.train_step(tra_batch, clean_batch)

            epoch_loss += loss

            if batch_idx % self.config['train']['log_interval'] == 0:
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Batch': f'{batch_idx + 1}/{self.train_batch_nums}'
                })

        avg_loss = epoch_loss / self.train_batch_nums

        print(f"Epoch {epoch} 训练完成，平均 Loss: {avg_loss:.4f}")
        return avg_loss

    def train_model(self, train_loader, val_loader, clean_loader, start_epoch=0, num_epochs=100, checkpoint_dir='./models/DITOL/checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_loss = float('inf')
        # 训练前预处理
        self.train_batch_nums = len(train_loader)
        self.val_batch_nums = len(val_loader)
        if self.config['train']['resume_training']:
            print("==> 尝试加载最近 checkpoint ...")
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path)
            else:
                print("未找到 checkpoint，重新训练。")

        for epoch in range(start_epoch, num_epochs):
            self.train_epoch(train_loader, clean_loader, epoch)

            # 验证集
            if val_loader:
                self.eval()
                val_stats = self.evaluate(val_loader, clean_loader)

                # 保存最优模型
                if val_stats['loss'] < best_loss:
                    best_loss = val_stats['loss']
                    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    self.save_checkpoint(epoch, best_path)
                    print(f"==> Best model saved to {best_path}")

            # 定期保存 checkpoint
            if (epoch + 1) % self.config['train']['checkpoint_save_interval'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch, checkpoint_path)
                print(f"==> Checkpoint saved to {checkpoint_path}")

        print("训练完成！")

    @torch.no_grad()
    def evaluate(self, val_loader,clean_loader):
        self.eval()
        total_loss = 0.0
        total_batches = len(val_loader)

        pbar = tqdm(zip(val_loader, clean_loader), total=total_batches, desc="Evaluating")
        for batch_idx, (val_batch, clean_batch) in enumerate(pbar):
            low_res_images, _, _ = process_batch(val_batch)
            low_res_images = low_res_images.to(self.config['device'])
            clean_image, _, _ = process_batch(clean_batch)
            clean_image = clean_image.to(self.config['device'])
            # 推理
            dehazed, _ = self.forward(low_res_images)

            # 计算 MSE 损失（你也可以添加感知损失等）
            loss = self.mse(dehazed, clean_image)
            total_loss += loss.item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Batch': f'{batch_idx + 1}/{total_batches}'
            })

        avg_loss = total_loss / total_batches
        print(f"验证完成，平均 Loss: {avg_loss:.4f}")

        return {
            'loss': avg_loss  # 保留字段名以兼容 train_model 中的逻辑
        }

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

    def predict(self, inputs):
        pass