import os

import torch
from torch import nn
from tqdm import tqdm

from utils.common import call_function
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FSDT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.val_batch_nums = None
        self.train_batch_nums = None
        self.cfg = cfg

        self.dehaze_name = self.cfg['method']['dehaze']
        self.detector_name = self.cfg['method']['detector']
        self.tracker_name = self.cfg['method']['tracker']
        self.detector_flag = cfg['detector_flag']
        self.tracker_flag = cfg['tracker_flag']
        self.freeze_dehaze = cfg['train']['freeze_dehaze']
        self.pretrain_flag = cfg['train']['pretrain_flag']
        
        
        self.dehaze = call_function(cfg['method']['dehaze'],
                                    f"models.dehaze.{cfg['method']['dehaze']}")

        self.detector = call_function(cfg['method']['detector'],
                                    f"models.detector.{cfg['method']['detector']}",cfg)

        self.tracker = call_function(cfg['method']['tracker'],
                                      f"models.trackers.{cfg['method']['tracker']}", cfg)

    def train_step(self,tra_batch, clean_batch):
        low_res_images, targets, ignore_list = tra_batch
        low_res_images = low_res_images.to(self.device)
        if clean_batch is not None:
            targets_img, _, _ = clean_batch
            targets_img = targets_img.to(self.device)
        else:
            targets_img = None
        self.optimizer.zero_grad()
        if self.cfg['train']['debug']:
            torch.autograd.set_detect_anomaly(True)
        if self.detector_flag:
            dehaze_imgs = self.dehaze(low_res_images)
            loss_dict = self.detector.forward_loss(dehaze_imgs, targets, ignore_list)
            loss_dict['total_loss'].backward()
        else:
            loss_dict = self.dehaze.forward_loss(low_res_images, targets_img)
            loss_dict['total_loss'].backward()
        self.optimizer.step()

        return loss_dict

    def train_epoch(self, train_loader, train_clean_loader, epoch):
        self.train()
        epoch_losses = {}
        if train_clean_loader is not None:
            loader = zip(train_loader, train_clean_loader)
        else:
            loader = ((tra_batch, None) for tra_batch in train_loader)

        pbar = tqdm(loader, total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, (tra_batch, clean_batch) in enumerate(pbar):
            loss_dict = self.train_step(tra_batch, clean_batch)

            # 累加每个 loss 项
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.detach().item()
                else:
                    epoch_losses[key] += float(value)
            if batch_idx % self.cfg['train']['log_interval'] == 0:
                postfix = {}
                for k, v in loss_dict.items():
                    try:
                        # 如果是 Tensor 且是标量，提取数值
                        if isinstance(v, torch.Tensor) and v.dim() == 0:
                            postfix[k] = f'{v.item():.4f}'
                        else:
                            postfix[k] = str(v)
                    except Exception as e:
                        postfix[k] = f'ERR({e})'
                postfix['Batch'] = f'{batch_idx + 1}/{self.train_batch_nums}'
                pbar.set_postfix(postfix)

        # 打印每个 loss 项的平均值
        print(f"Epoch {epoch} 训练完成，平均 Loss:")
        for key, total in epoch_losses.items():
            avg = total / self.train_batch_nums
            print(f"  {key}: {avg:.4f}")

    def train_model(self, train_loader, val_loader, train_clean_loader, val_clean_loader, num_epochs=100):
        # 训练前预处理
        self.train_batch_nums = len(train_loader)
        self.val_batch_nums = len(val_loader)
        best_loss = float('inf')
        dehaze_ckpt = f'models/dehaze/{self.dehaze_name}/ckpt'
        detector_ckpt = f'models/detector/{self.detector_name}/ckpt'
        combo_ckpt_model = os.path.join(
            f'models/dehaze/{self.dehaze_name}/ckpt',
            f"{self.dehaze_name}_{self.detector_name}_ckpt_epoch_0.pth"
        )
        if os.path.exists(combo_ckpt_model):
            print(f"==> 加载组合模型权重：{combo_ckpt_model}")
            self.dehaze.load_state_dict(torch.load(combo_ckpt_model, map_location=self.device))
            detector_combo_ckpt = os.path.join(
                f'models/detector/{self.detector_name}/ckpt',
                f"{self.dehaze_name}_{self.detector_name}_ckpt_epoch_0.pth"
            )
            if os.path.exists(detector_combo_ckpt):
                self.detector.load_state_dict(torch.load(detector_combo_ckpt, map_location=self.device))
                print(f"==> 加载检测器组合模型权重：{detector_combo_ckpt}")
            else:
                print("==> 未找到检测器组合模型，仅加载去雾模型")
        if self.cfg['train']['resume_training']:
            # 检测是否加载成功
            dehaze_loaded = False
            detector_loaded = False
            print("==> 尝试加载模型继续训练 ...")
            dehaze_ckpt_model = os.path.join(f'models/dehaze/{self.dehaze_name}/ckpt', 'pretrain.pth')
            dehaze_detector_model = os.path.join(f'models/dehaze/{self.dehaze_name}/ckpt',
                                                 f"{self.detector_name}_pretrain.pth")
            detector_ckpt_model = os.path.join(f'models/detector/{self.detector_name}/ckpt', 'best.pth')

            if os.path.exists(dehaze_ckpt_model):
                self.dehaze.load_state_dict(torch.load(dehaze_ckpt_model))
                dehaze_loaded = True
            elif os.path.exists(dehaze_detector_model):
                self.detector.load_state_dict(torch.load(dehaze_detector_model))
                dehaze_loaded = True
            else:
                print("未找到预训练去雾模型，重新训练。")
            if os.path.exists(detector_ckpt_model):
                self.detector.load_state_dict(torch.load(detector_ckpt_model))
                detector_loaded = True
            else:
                print("未找到已有检测模型，重新训练。")
            if (dehaze_loaded and detector_loaded) or dehaze_loaded:
                best_loss = self.evaluate(val_loader, val_clean_loader)

        if self.freeze_dehaze:
            for param in self.dehaze.parameters():
                param.requires_grad = False
            print("冻结预训练去雾模型")
        elif self.pretrain_flag:
            self.detector_flag = False

        for epoch in range(0, num_epochs):
            if hasattr(self.detector, '_use_mse'):
                self.detector._use_mse = (epoch < 2)  # 训练前两轮使用 MSE
            # pretrain
            if not self.freeze_dehaze and epoch > self.cfg['train']['dehaze_epoch'] and self.pretrain_flag:
                self.detector_flag = True
                self.freeze_dehaze = True
                best_loss = float('inf')
                for param in self.dehaze.parameters():
                    param.requires_grad = False
                print("冻结预训练去雾模型")
            self.train_epoch(train_loader, train_clean_loader, epoch)
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1} finished. Current LR: {current_lr:.6f}")
            # 验证
            val_loss = self.evaluate(val_loader, val_clean_loader)
            # 如果检测器效果比历史好
            if val_loss < best_loss:
                # 如果此时检测器在训练
                if self.detector_flag:
                    best_loss = val_loss
                    self.save_model(self.detector,detector_ckpt, self.detector_name,f"best.pth")
                    # 如果同时不冻结去雾模型
                    if not self.freeze_dehaze:
                        self.save_model(self.dehaze, dehaze_ckpt, self.dehaze_name,
                                        f"{self.detector_name}_pretrain.pth")
                # 此时去雾模块在训练
                elif not self.freeze_dehaze:
                    best_loss = val_loss
                    self.save_model(self.dehaze,dehaze_ckpt, self.dehaze_name,f"pretrain.pth")
            # 每checkpoint_interval存储一次模型
            if epoch % self.cfg['train']['checkpoint_interval'] == 0 and self.detector_flag:
                # 保存一张预览图

                self.save_model(self.detector, detector_ckpt, self.detector_name,
                                f"{self.dehaze_name}_{self.detector_name}_ckpt_epoch_{epoch + 1}.pth")
                if not self.freeze_dehaze:
                    self.save_model(self.dehaze, dehaze_ckpt, self.dehaze_name,
                                    f"{self.dehaze_name}_{self.detector_name}_ckpt_epoch_{epoch + 1}.pth")



    @torch.no_grad()
    def evaluate(self, val_loader, val_clean_loader):
        self.eval()
        epoch_losses = {}
        if val_clean_loader is not None:
            loader = zip(val_loader, val_clean_loader)
        else:
            loader = ((tra_batch, None) for tra_batch in val_loader)
        pbar = tqdm(loader, total=self.val_batch_nums, desc=f"val")
        for batch_idx, (val_batch, clean_batch) in enumerate(pbar):
            low_res_images, targets, ignore_list = val_batch
            low_res_images = low_res_images.to(self.device)
            if clean_batch is not None:
                targets_img, _, _ = clean_batch
                targets_img = targets_img.to(self.device)
            else:
                targets_img = None
            if self.detector_flag:
                dehaze_imgs = self.dehaze(low_res_images)
                loss_dict = self.detector.forward_loss(dehaze_imgs, targets, ignore_list)
            else:
                loss_dict = self.dehaze.forward_loss(low_res_images, targets_img)
            # 累加每个 loss 项
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.detach().item()
                else:
                    epoch_losses[key] += float(value)

            if batch_idx % self.cfg['train']['log_interval'] == 0:
                postfix = {}
                for k, v in loss_dict.items():
                    try:
                        # 如果是 Tensor 且是标量，提取数值
                        if isinstance(v, torch.Tensor) and v.dim() == 0:
                            postfix[k] = f'{v.item():.4f}'
                        else:
                            postfix[k] = str(v)
                    except Exception as e:
                        postfix[k] = f'ERR({e})'
                postfix['Batch'] = f'{batch_idx + 1}/{self.train_batch_nums}'
                pbar.set_postfix(postfix)

        # 打印每个 loss 项的平均值
        print(f"验证完成，平均 Loss: {epoch_losses['total_loss']/ self.val_batch_nums:.4f}")
        for key, total in epoch_losses.items():
            avg = total / self.val_batch_nums
            print(f"  {key}: {avg:.4f}")
            epoch_losses[key] = avg
        return epoch_losses['total_loss']/ self.val_batch_nums

    def predict(self, x):
        # 1. 图像去雾
        x = self.dehaze(x)
        if self.detector_flag:
            img = x
            # 2. 目标检测
            x = self.detector.predict(img, conf_thresh=self.cfg['conf_threshold'])
            if self.tracker_flag:
                x = [det + [0] if len(det) == 5 else det for det in x]
                # 3. 目标跟踪
                x = self.tracker.update(x, img, None)
        return x

    def save_model(self, component, save_path, model_name, save_name):
        save_dir = os.path.join("", save_path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{save_name}")
        torch.save(component.state_dict(), save_path)
        print(f"{model_name} 模块已保存到: {save_path}")

    def load_model(self):
        print("==> 尝试加载模型继续训练 ...")
        dehaze_ckpt_model = os.path.join(f'models/dehaze/{self.dehaze_name}/ckpt', 'pretrain.pth')
        dehaze_detector_model = os.path.join(f'models/dehaze/{self.dehaze_name}/ckpt',
                                             f"{self.detector_name}_pretrain.pth")
        detector_ckpt_model = os.path.join(f'models/detector/{self.detector_name}/ckpt', 'best.pth')

        if os.path.exists(dehaze_ckpt_model) and not self.detector_flag:
            self.dehaze.load_state_dict(torch.load(dehaze_ckpt_model))
            print('去雾模型加载成功')
        elif self.detector_flag:
            if os.path.exists(dehaze_detector_model):
                self.dehaze.load_state_dict(torch.load(dehaze_detector_model))
                print('去雾_检测模型加载成功')
            if os.path.exists(detector_ckpt_model):
                self.detector.load_state_dict(torch.load(detector_ckpt_model))
                print('检测模型加载成功')

