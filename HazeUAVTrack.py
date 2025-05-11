import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.common import call_function

# HazeUAVTrack 类：集成去雾、目标检测和目标跟踪功能的端到端模型
# 继承自 nn.Module，可以在 PyTorch 框架下进行训练和推理
class HazeUAVTrack(nn.Module):
    def __init__(self, cfg):
        """
        初始化 HazeUAVTrack 模型。
        根据配置文件加载并初始化去雾模块、检测模块和跟踪模块。
        同时读取训练和推理相关的配置参数，如阈值、训练标志等。
        """
        super().__init__()
        self.val_batch_nums = None
        self.train_batch_nums = None
        self.cfg = cfg

        # 从配置中获取并保存各个模块的名称及启用标志
        self.dehaze_name = self.cfg['method']['dehaze']
        self.detector_name = self.cfg['method']['detector']
        self.tracker_name = self.cfg['method']['tracker']
        self.detector_flag = cfg['detector_flag']
        self.tracker_flag = cfg['tracker_flag']
        self.freeze_dehaze = cfg['train']['freeze_dehaze'] # 是否冻结去雾模块 (训练时)
        self.pretrain_flag = cfg['train']['pretrain_flag'] # 是否进行预训练流程 (先训去雾)
        self.conf_thresh = self.cfg['conf_threshold'] # 检测置信度阈值
        self.iou_thresh = self.cfg['iou_threshold'] # 检测 NMS 的 IoU 阈值

        # 动态加载并实例化去雾模型
        self.dehaze = call_function(cfg['method']['dehaze'],
                                    f"models.dehaze.{cfg['method']['dehaze']}")

        # 动态加载并实例化检测模型，并将完整配置 cfg 传递给检测器
        self.detector = call_function(cfg['method']['detector'],
                                    f"models.detector.{cfg['method']['detector']}",cfg)

        # 动态加载并实例化跟踪模型，并将完整配置 cfg 传递给跟踪器
        self.tracker = call_function(cfg['method']['tracker'],
                                      f"models.trackers.{cfg['method']['tracker']}", cfg)

        # 注意：self.device, self.optimizer, self.scheduler 需要在外部 (如训练脚本) 设置

    def train_step(self,tra_batch, clean_batch):
        """
        执行单个训练批次的向前传播、损失计算和反向传播。
        根据 self.detector_flag 和 self.freeze_dehaze 决定是训练去雾模块还是检测模块。
        如果训练检测器，先通过去雾模块处理图像。
        """
        # 将数据移动到指定设备
        low_res_images, targets, ignore_list = tra_batch
        low_res_images = low_res_images.to(self.device)
        # 处理干净图像数据 (用于去雾训练)
        if clean_batch is not None:
            targets_img, _, _ = clean_batch
            targets_img = targets_img.to(self.device)
        else:
            targets_img = None

        # 清零梯度
        self.optimizer.zero_grad()
        # 开启异常检测 (可选，用于调试)
        if self.cfg['train']['debug']:
            torch.autograd.set_detect_anomaly(True)

        # 根据训练目标执行不同的前向和损失计算
        if self.detector_flag:
            # 训练检测器：先去雾，再计算检测损失
            dehaze_imgs = self.dehaze(low_res_images)
            loss_dict = self.detector.forward_loss(dehaze_imgs, targets, ignore_list)
            loss_dict['total_loss'].backward() # 对总损失进行反向传播
        else:
            # 训练去雾器：计算去雾损失
            loss_dict = self.dehaze.forward_loss(low_res_images, targets_img)
            loss_dict['total_loss'].backward() # 对总损失进行反向传播

        # 执行优化步骤
        self.optimizer.step()

        return loss_dict # 返回当前批次的损失信息

    def train_epoch(self, train_loader, train_clean_loader, epoch):
        """
        执行一个完整的训练周期 (epoch)。
        遍历训练数据加载器，调用 train_step 处理每个批次。
        累加并打印 epoch 级别的平均损失。
        """
        self.train() # 设置模型为训练模式
        epoch_losses = {} # 存储整个 epoch 的累积损失
        # 根据是否有干净图像加载器构建数据迭代器
        if train_clean_loader is not None:
            loader = zip(train_loader, train_clean_loader)
        else:
            loader = ((tra_batch, None) for tra_batch in train_loader)

        # 使用 tqdm 显示训练进度条
        pbar = tqdm(loader, total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, (tra_batch, clean_batch) in enumerate(pbar):
            # 执行单个训练批次的处理
            loss_dict = self.train_step(tra_batch, clean_batch)

            # 累加当前批次的损失到 epoch 总损失中
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                # 确保累加的是数值类型
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.detach().item()
                else:
                    epoch_losses[key] += float(value)

            # 每隔一定批次打印当前损失信息到进度条后缀
            if batch_idx % self.cfg['train']['log_interval'] == 0:
                postfix = {}
                for k, v in loss_dict.items():
                    try:
                        if isinstance(v, torch.Tensor) and v.dim() == 0:
                            postfix[k] = f'{v.item():.4f}'
                        else:
                            postfix[k] = str(v)
                    except Exception as e:
                        postfix[k] = f'ERR({e})'
                postfix['Batch'] = f'{batch_idx + 1}/{self.train_batch_nums}'
                pbar.set_postfix(postfix)

        # epoch 结束后，计算并打印平均损失
        print(f"Epoch {epoch} 训练完成，平均 Loss:")
        for key, total in epoch_losses.items():
            avg = total / self.train_batch_nums
            print(f"  {key}: {avg:.4f}")

    def train_model(self, train_loader, val_loader, train_clean_loader, val_clean_loader, num_epochs=100):
        """
        管理整个训练过程的主函数。
        处理模型加载、不同训练阶段的切换 (预训练去雾 -> 训练检测器)、
        每个 epoch 的训练和验证、学习率调度以及模型保存。
        """
        # 根据配置生成用于文件名的雾强度字符串
        fog_strength_str = f"fog_{int(self.cfg['dataset']['fog_strength'] * 100):03d}"
        self.train_batch_nums = len(train_loader)
        self.val_batch_nums = len(val_loader)
        best_loss = float('inf') # 记录最佳验证损失，用于保存模型

        # 定义模型保存路径
        dehaze_ckpt = f'models/dehaze/{self.dehaze_name}/ckpt'
        detector_ckpt = f'models/detector/{self.detector_name}/ckpt'

        # 处理断点续训逻辑：尝试加载之前保存的模型权重
        if self.cfg['train']['resume_training']:
            dehaze_loaded = False
            detector_loaded = False
            print("==> 尝试加载模型继续训练 ...")
            # 构造预训练去雾模型和最佳检测模型的路径
            dehaze_ckpt_model = os.path.join(dehaze_ckpt, f'{fog_strength_str}_pretrain.pth')
            detector_ckpt_model = os.path.join(detector_ckpt, f'{fog_strength_str}_best.pth')

            # 尝试加载模型权重并更新加载状态
            if os.path.exists(dehaze_ckpt_model):
                 try:
                     self.dehaze.load_state_dict(torch.load(dehaze_ckpt_model, map_location=self.device))
                     dehaze_loaded = True
                     print(f"成功加载去雾模型: {dehaze_ckpt_model}")
                 except Exception as e:
                     print(f"加载去雾模型失败: {e}")
            else:
                print("未找到预训练去雾模型文件。")

            if os.path.exists(detector_ckpt_model):
                try:
                    self.detector.load_state_dict(torch.load(detector_ckpt_model, map_location=self.device))
                    detector_loaded = True
                    print(f"成功加载检测模型: {detector_ckpt_model}")
                except Exception as e:
                    print(f"加载检测模型失败: {e}")
            else:
                print("未找到已有检测模型文件。")

            # 如果成功加载了至少一个相关模型，进行一次评估以初始化最佳损失
            if ((dehaze_loaded and detector_loaded) or
                    (dehaze_loaded and not self.freeze_dehaze and self.pretrain_flag) or # 如果在去雾预训练阶段且加载了去雾模型
                    (self.freeze_dehaze and detector_loaded) or # 如果冻结去雾训练检测器且加载了检测器模型
                    (self.detector_flag and detector_loaded) ): # 如果正在训练检测器且加载了检测器模型
                print("加载成功，进行初始评估...")
                best_loss = self.evaluate(val_loader, val_clean_loader)
                print(f"初始验证损失: {best_loss:.4f}")
            else:
                 print("未加载到足够的模型权重，从头开始训练。")


        # 根据配置决定是否冻结去雾模块
        if self.freeze_dehaze:
            for param in self.dehaze.parameters():
                param.requires_grad = False
            print("根据配置冻结去雾模型权重")
        # 如果设置了预训练标志且去雾模块未被冻结，则先进入去雾预训练阶段
        elif self.pretrain_flag:
            self.detector_flag = False # 暂时关闭检测器训练，只训练去雾模块
            print("进入去雾模块预训练阶段...")


        # 开始主训练循环
        for epoch in range(0, num_epochs):
            # 如果检测器支持，可能在训练初期切换损失函数类型 (例如，YOLO 系列的边界框损失)
            if hasattr(self.detector, '_use_mse'):
                self.detector._use_mse = (epoch < 4) # 示例：前 4 个 epoch 使用 MSE

            # 处理去雾预训练阶段结束后的切换逻辑
            # 当达到指定的去雾预训练 epoch 数，且当前正在进行去雾预训练时，切换到检测器训练
            if not self.freeze_dehaze and epoch >= self.cfg['train']['dehaze_epoch'] and self.pretrain_flag:
                print(f"去雾预训练阶段结束 (epoch {epoch})，切换到检测器训练阶段。")
                self.detector_flag = True # 开启检测器训练
                self.freeze_dehaze = True # 冻结去雾模块
                best_loss = float('inf') # 重置最佳损失，以便保存检测器模型
                for param in self.dehaze.parameters():
                    param.requires_grad = False # 确保去雾模块被冻结
                print("冻结预训练去雾模型权重")

            # 执行当前 epoch 的训练
            self.train_epoch(train_loader, train_clean_loader, epoch)

            # 如果存在学习率调度器，更新学习率
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1} finished. Current LR: {current_lr:.6f}")

            # 在验证集上评估模型性能
            val_loss = self.evaluate(val_loader, val_clean_loader)

            # 根据验证损失决定是否保存模型
            if val_loss < best_loss:
                # 如果当前正在训练检测器，且验证损失更好，保存最佳检测器模型
                if self.detector_flag:
                    best_loss = val_loss
                    self.save_model(self.detector, detector_ckpt, self.detector_name, f"{fog_strength_str}_best.pth")
                # 如果当前正在训练去雾器 (且未冻结)，且验证损失更好，保存最佳去雾模型 (作为预训练模型)
                elif not self.freeze_dehaze:
                    best_loss = val_loss
                    self.save_model(self.dehaze, dehaze_ckpt, self.dehaze_name, f"{fog_strength_str}_pretrain.pth")

            # 每隔 checkpoint_interval 个 epoch 保存一次检查点 (主要保存检测器，去雾器可选)
            if epoch % self.cfg['train']['checkpoint_interval'] == 0 and self.detector_flag:
                # 保存检测器模型的当前状态
                self.save_model(self.detector, detector_ckpt, self.detector_name,
                                f"{self.dehaze_name}_{self.detector_name}_ckpt_epoch_{epoch + 1}.pth")
                # 如果去雾器也在训练，也可以选择保存其检查点
                if not self.freeze_dehaze:
                     self.save_model(self.dehaze, dehaze_ckpt, self.dehaze_name,
                                     f"{self.dehaze_name}_{self.detector_name}_dehaze_ckpt_epoch_{epoch + 1}.pth")


    @torch.no_grad() # 禁用梯度计算，用于评估模式
    def evaluate(self, val_loader, val_clean_loader):
        """
        在验证集上评估模型的性能，计算并返回平均损失。
        """
        self.eval() # 设置模型为评估模式
        # 如果检测器有评估模式下的特殊处理，设置标志
        if hasattr(self.detector, '_evaluating'):
            self.detector._evaluating = True

        epoch_losses = {} # 存储整个验证过程的累积损失
        # 根据是否有干净图像加载器构建数据迭代器
        if val_clean_loader is not None:
            loader = zip(val_loader, val_clean_loader)
        else:
            loader = ((val_batch, None) for val_batch in val_loader)

        # 使用 tqdm 显示验证进度条
        pbar = tqdm(loader, total=self.val_batch_nums, desc=f"val")
        for batch_idx, (val_batch, clean_batch) in enumerate(pbar):
            # 将数据移动到指定设备
            low_res_images, targets, ignore_list = val_batch
            low_res_images = low_res_images.to(self.device)
            # 处理干净图像数据
            if clean_batch is not None:
                targets_img, _, _ = clean_batch
                targets_img = targets_img.to(self.device)
            else:
                targets_img = None

            # 根据评估目标执行不同的前向和损失计算
            if self.detector_flag:
                # 评估检测器：先去雾，再计算检测损失
                dehaze_imgs = self.dehaze(low_res_images)
                loss_dict = self.detector.forward_loss(dehaze_imgs, targets, ignore_list)
            else:
                # 评估去雾器：计算去雾损失
                loss_dict = self.dehaze.forward_loss(low_res_images, targets_img)

            # 累加当前批次的损失
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.detach().item()
                else:
                    epoch_losses[key] += float(value)

            # 定期更新进度条的后缀信息
            if batch_idx % self.cfg['train']['log_interval'] == 0:
                postfix = {}
                for k, v in loss_dict.items():
                    try:
                        if isinstance(v, torch.Tensor) and v.dim() == 0:
                            postfix[k] = f'{v.item():.4f}'
                        else:
                            postfix[k] = str(v)
                    except Exception as e:
                        postfix[k] = f'ERR({e})'
                postfix['Batch'] = f'{batch_idx + 1}/{self.val_batch_nums}'
                pbar.set_postfix(postfix)

        # 验证结束后，计算并打印平均损失
        total_avg_loss = epoch_losses['total_loss'] / self.val_batch_nums
        print(f"验证完成，平均 Loss: {total_avg_loss:.4f}")
        for key, total in epoch_losses.items():
            avg = total / self.val_batch_nums
            print(f"  {key}: {avg:.4f}")
            epoch_losses[key] = avg # 将平均值存回字典

        # 重置评估模式标志
        if hasattr(self.detector, '_evaluating'):
            self.detector._evaluating = False

        return total_avg_loss # 返回总的平均验证损失

    def predict(self, x):
        """
        执行端到端的推理过程：输入雾图 -> 去雾 -> (如果启用检测器) -> 检测 -> (如果启用跟踪器) -> 跟踪。
        返回最终的检测或跟踪结果。
        """
        # 1. 图像去雾
        x = self.dehaze(x)

        # 如果启用了检测器，则进行后续的检测和跟踪
        if self.detector_flag:
            img = x # 去雾后的图像用于检测和跟踪
            # 2. 目标检测：使用检测器在去雾后的图像上进行预测
            # 使用配置的置信度和 IoU 阈值过滤结果
            x = self.detector.predict(img, conf_thresh=self.conf_thresh, iou_thresh=self.conf_thresh)

            # 如果启用了跟踪器，则进行跟踪
            if self.tracker_flag:
                # 将检测器的输出格式转换为跟踪器所需的 NumPy 数组格式
                detections_for_tracker = None
                if isinstance(x, list): # YOLOv5 等可能返回列表
                    if not x:
                        # 如果检测结果列表为空，创建空的 NumPy 数组
                        num_detection_features = 6 # 假设每行格式是 [x1, y1, x2, y2, conf, class_id]
                        detections_for_tracker = np.empty((0, num_detection_features))
                    else:
                        # 合并列表中的 Tensor，移到 CPU，转为 NumPy
                        tensors_on_cpu = [t.cpu() for t in x]
                        stacked_tensor = torch.cat(tensors_on_cpu, dim=0)
                        detections_for_tracker = stacked_tensor.numpy()
                elif isinstance(x, torch.Tensor): # 其他检测器可能直接返回 Tensor
                    detections_for_tracker = x.cpu().numpy()
                elif isinstance(x, np.ndarray): # 如果已经是 NumPy 数组
                    detections_for_tracker = x
                else:
                    print(f"Warning: Unexpected input type for tracker: {type(x)}")
                    num_detection_features = 6
                    detections_for_tracker = np.empty((0, num_detection_features))

                # 3. 目标跟踪：使用跟踪器更新并获取跟踪结果
                # 将检测结果和当前帧图像传递给跟踪器
                x = self.tracker.update(detections_for_tracker, img, None) # None 表示没有额外的输入 (如光流)

        return x # 返回最终的预测结果 (检测框或跟踪轨迹)

    def save_model(self, component, save_path, model_name, save_name):
        """
        保存指定模块 (去雾器或检测器) 的模型权重到文件。
        """
        save_dir = os.path.join("", save_path) # 构建保存目录路径
        os.makedirs(save_dir, exist_ok=True) # 创建目录（如果不存在）
        # 构建完整的保存文件路径
        save_path = os.path.join(save_dir, f"{save_name}")
        # 保存模型的 state_dict (模型参数)
        torch.save(component.state_dict(), save_path)
        print(f"{model_name} 模块已保存到: {save_path}")

    def load_model(self):
        """
        根据配置和标志加载模型权重 (用于推理或非 resume 方式的加载)。
        主要用于加载预训练去雾模型或最佳检测模型。
        """
        # 根据雾强度构建文件名
        fog_strength_str = f"fog_{int(self.cfg['dataset']['fog_strength'] * 100):03d}"
        print("==> 尝试加载模型 ...")
        # 构造预训练去雾模型和最佳检测模型的默认加载路径
        dehaze_ckpt_model = os.path.join(f'models/dehaze/{self.dehaze_name}/ckpt', f"{fog_strength_str}_pretrain.pth")
        detector_ckpt_model = os.path.join(f'models/detector/{self.detector_name}/ckpt', f"{fog_strength_str}_best.pth")
        print(f"去雾模型加载路径: {dehaze_ckpt_model}")
        print(f"检测模型加载路径: {detector_ckpt_model}")

        # 根据当前模型的使用模式 (是否启用检测器) 选择加载哪个模块
        # 如果未启用检测器，尝试加载去雾模型
        if not self.detector_flag:
             if os.path.exists(dehaze_ckpt_model):
                try:
                    # 加载去雾模型的 state_dict
                    self.dehaze.load_state_dict(torch.load(dehaze_ckpt_model, map_location=self.device))
                    print('去雾模型加载成功')
                except Exception as e:
                    print(f"去雾模型加载失败: {e}")
                    print("未加载去雾模型。")
             else:
                 print("未找到去雾模型文件，未加载。")
        # 如果启用了检测器，尝试加载检测模型
        elif self.detector_flag:
            if os.path.exists(detector_ckpt_model):
                try:
                    # 加载检测模型的 state_dict
                    self.detector.load_state_dict(torch.load(detector_ckpt_model, map_location=self.device))
                    print('检测模型加载成功')
                except Exception as e:
                    print(f"检测模型加载失败: {e}")
                    print("未加载检测模型。")
            else:
                print("未找到检测模型文件，未加载。")
