import os

import cv2
import torch
import torch.nn as nn
import ultralytics
from tqdm import tqdm

from ultralytics import YOLO

from .yolo_utils import process_and_return_loaders, changeed__call__


# --- 保留的图像增强模块 ---

# Lap_Pyramid_Conv 类：用于构建拉普拉斯金字塔和重建
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        # 使用 OpenCV 生成高斯核
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        # 使用反射填充避免边界效应
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        # 创建一个两倍大小的零张量
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        # 将输入复制到偶数位置，并乘以4（为了保持能量）
        up[:, :, ::2, ::2] = x * 4
        # 应用高斯滤波
        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # 将高斯核移动到输入图像相同的设备
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        # 构建拉普拉斯金字塔的高频层和最低频层（残差）
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up # 高频信息
            pyr.append(diff)
            current = down # 最低频层
        pyr.append(current) # 添加最低频层
        # 返回的 pyr 是 [高频1, 高频2, ..., 高频N, 最低频]
        return pyr

    def pyramid_recons(self, pyr):
        # 从最低频层开始重建图像
        image = pyr[-1] # 最低频层是最后一个
        # 从倒数第二个元素（最高频层）开始迭代
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            image = up + level # 加上对应的高频信息
        return image

# ChannelAttention 类：通道注意力模块 (虽然未直接被 DENet 使用，但保留以防万一)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的多层感知机 (MLP)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x # 注意力权重乘以输入

# SpatialAttention 类：空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7): # 原始代码是5，这里改为7，更常见
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        # 对平均池化和最大池化的结果（堆叠后是2通道）应用卷积
        self.conv = nn.Conv2d(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的平均和最大值，保持维度
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 沿通道维度堆叠
        attention = torch.cat([avgout, maxout], dim=1)
        # 应用卷积和 sigmoid 获取空间注意力图
        attention = self.conv(attention)
        return self.sigmoid(attention) * x # 注意力权重乘以输入

# Trans_guide 类：用于生成指导信息
class Trans_guide(nn.Module):
    def __init__(self, ch=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1), # 输入是原始图像(3)和低频增强结果(3)，共6通道
            nn.LeakyReLU(True),
            SpatialAttention(3), # 应用空间注意力
            nn.Conv2d(ch, 3, 3, padding=1), # 输出3通道的指导信息
        )

    def forward(self, x): # x 是 torch.cat([原始图像, 低频增强结果], dim=1)
        return self.layer(x)

# Trans_low 类：处理拉普拉斯金字塔的最低频层
class Trans_low(nn.Module):
    def __init__(
        self,
        ch_blocks=64, # 中间特征通道数
        ch_mask=16, # 指导信息模块的中间通道数
    ):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        # 多尺度卷积 (MM) 部分
        # 修正：原代码是4个mm1，可能意图是多尺度，这里改为不同大小的卷积核
        self.mm1 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=1,
                             padding=0)
        self.mm2 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=3,
                             padding=3 // 2)
        self.mm3 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=5,
                             padding=5 // 2)
        self.mm4 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=7,
                             padding=7 // 2)


        # 解码器
        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1))

        # 指导信息生成模块
        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x): # x 是拉普拉斯金字塔的最低频层
        x1 = self.encoder(x)
        # 应用多尺度卷积并拼接
        x1_1 = self.mm1(x1)
        x1_2 = self.mm2(x1)
        x1_3 = self.mm3(x1)
        x1_4 = self.mm4(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1) # 拼接回 ch_blocks 通道
        x1 = self.decoder(x1) # 解码回 3 通道

        out = x + x1 # 残差连接
        out = torch.relu(out) # 应用 ReLU 激活

        # 生成指导信息
        mask = self.trans_guide(torch.cat([x, out], dim=1))
        return out, mask # 返回增强后的低频层和指导信息

# SFT_layer 类：空间特征变换 (Spatial Feature Transform) 层
class SFT_layer(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()

        # 主分支：对输入应用卷积和激活
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        # 主分支：解码回输出通道数
        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))

        # 根据指导信息预测 shift 参数
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        # 根据指导信息预测 scale 参数
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, guide): # x 是拉普拉斯金字塔的高频层，guide 是指导信息
        processed_x = self.encoder(x)
        scale = self.scale_conv(guide)
        shift = self.shift_conv(guide)
        # 应用 SFT 变换: x = scale * x + shift
        processed_x = processed_x + processed_x * scale + shift
        processed_x = self.decoder(processed_x)
        return processed_x

# Trans_high 类：处理拉普拉斯金字塔的高频层
class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()

        # 使用 SFT 层进行高频增强
        self.sft = SFT_layer(in_ch, inter_ch, out_ch, kernel_size)

    def forward(self, x, guide): # x 是拉普拉斯金字塔的高频层，guide 是指导信息
        # 残差连接：原始高频层 + SFT 增强后的高频层
        return x + self.sft(x, guide)

# Up_guide 类：上采样指导信息
class Up_guide(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            # 双线性插值上采样
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # 卷积调整通道或平滑
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x): # x 是低一层的指导信息
        x = self.up(x)
        return x

# DENet 类：完整的图像增强网络
class DENet(nn.Module):
    def __init__(self,
                 num_high=3, # 拉普拉斯金字塔的高频层数
                 ch_blocks=64, # Trans_low 的中间通道数
                 up_ksize=1, # Up_guide 的卷积核大小
                 high_ch=32, # Trans_high (SFT_layer) 的中间通道数
                 high_ksize=3, # Trans_high (SFT_layer) 的卷积核大小
                 ch_mask=16, # Trans_guide 的中间通道数
                 gauss_kernel=5): # 拉普拉斯金字塔的高斯核大小
        super().__init__()
        self.num_high = num_high
        # 拉普拉斯金字塔分解/重建模块
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        # 处理最低频层的模块
        self.trans_low = Trans_low(ch_blocks, ch_mask)

        # 为每个高频层创建上采样指导模块和高频处理模块
        for i in range(0, self.num_high):
            self.__setattr__('up_guide_layer_{}'.format(i),
                             Up_guide(up_ksize, ch=3)) # 指导信息通道数应与图像通道数一致 (3)
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize)) # 输入输出都是3通道

    def forward(self, x): # x 是原始输入图像
        # 拉普拉斯金字塔分解
        pyrs = self.lap_pyramid.pyramid_decom(img=x) # pyrs = [高频1, ..., 高频N, 最低频]

        trans_pyrs = [] # 存储增强后的金字塔层

        # 处理最低频层
        trans_pyr, guide = self.trans_low(pyrs[-1]) # pyrs[-1] 是最低频层
        trans_pyrs.append(trans_pyr) # 将增强后的最低频层添加到结果列表

        # 上采样指导信息到每个高频层对应的分辨率
        commom_guide = []
        current_guide = guide # 从最低频层的指导信息开始
        for i in range(self.num_high):
            # 逐层上采样指导信息
            current_guide = self.__getattr__('up_guide_layer_{}'.format(i))(current_guide)
            commom_guide.append(current_guide) # 存储上采样后的指导信息

        # 处理高频层（从低分辨率高频到高分辨率高频）
        # 注意 pyrs 是 [高频1(高分), ..., 高频N(低分), 最低频(最低分)]
        # 处理顺序应该是从低分高频到高分高频，对应 pyrs[-2], pyrs[-3], ..., pyrs[-num_high-1]
        # 对应 commom_guide[0], commom_guide[1], ..., commom_guide[num_high-1]
        for i in range(self.num_high):
            # pyrs[-2-i] 是对应的高频层
            # commom_guide[i] 是对应分辨率的指导信息
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_guide[i])
            trans_pyrs.append(trans_pyr) # 将增强后的高频层添加到结果列表

        # trans_pyrs 现在是 [增强最低频, 增强高频N, 增强高频N-1, ..., 增强高频1]
        # 重建图像需要金字塔层按 [增强高频1, ..., 增强高频N, 增强最低频] 的顺序
        # 将 trans_pyrs 顺序调整为从高频到低频
        trans_pyrs.reverse() # 现在是 [增强高频1, ..., 增强高频N, 增强最低频]

        # 重建增强后的图像
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out


class DE_NET(nn.Module):
    def __init__(self,
                 cfg,
                 yolo_cfg='yolov3.yaml', # ultralytics 模型配置文件
                 yolo_weights='yolov3u.pt'): # ultralytics 模型权重文件
        super().__init__()
        self.config = cfg
        # 图像增强模块
        self.enhancement = DENet()

        # 使用 ultralytics 加载 YOLOv3 模型
        # yolov3_wrapper 是 ultralytics 的 YOLO 对象，提供了train/val/predict等方法
        print("Loading YOLOv3u model using ultralytics.YOLO...")
        self.yolov3_wrapper = YOLO(yolo_cfg)
        # .model 属性是底层的 PyTorch nn.Module，用于直接的前向传播
        self.yolov3 = self.yolov3_wrapper.load(yolo_weights).model
        ultralytics.utils.loss.v8DetectionLoss.__call__ = changeed__call__

    def forward(self, x):
        # 1. 应用图像增强
        x_enhanced = self.enhancement(x)

        yolo_out = self.yolov3(x_enhanced)

        # 返回 ultralytics YOLO 模型的输出
        return yolo_out


    def freeze_parms(self, enhancement=False, yolov3=True):
        """
        冻结增强模块或加载的 YOLOv3 模型的参数。
        默认冻结 YOLOv3 部分。
        """
        if enhancement:
             print("冻结增强模块参数...")
             for param in self.enhancement.parameters():
                 param.requires_grad = False
        if yolov3:
             print("冻结加载的 YOLOv3 模型参数...")
             for param in self.yolov3.parameters():
                 param.requires_grad = False

    def unfreeze_parms(self, enhancement=False, yolov3=True):
        """
        解冻增强模块或加载的 YOLOv3 模型的参数。
        默认解冻 YOLOv3 部分。
        """
        if enhancement:
             print("解冻增强模块参数...")
             for param in self.enhancement.parameters():
                 param.requires_grad = True
        if yolov3:
             print("解冻加载的 YOLOv3 模型参数...")
             for param in self.yolov3.parameters():
                 param.requires_grad = True

    def model_info(self, verbose=False):
        """
        打印模型信息，包括增强模块和加载的 YOLO 模型。
        """
        print("\n--- DENet (Enhancement Module) Info ---")
        print(self.enhancement) # 简单打印模块结构

        print("\n--- Loaded ultralytics YOLO Model Info ---")
        if hasattr(self.yolov3_wrapper, 'info'):
             # ultralytics 的 YOLO 对象通常有 info 方法
             self.yolov3_wrapper.info()
        elif hasattr(self.yolov3, 'info'):
             # 底层模型可能有 info 方法
             self.yolov3.info()
        else:
             print("加载的 YOLO 模型没有 'info()' 方法可供调用。")

    def train_step(self, tra_batch):
        self.optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)

        low_res_images, targets_yolov3, processed_ignore_list = tra_batch
        # print(type(targets_yolov3))
        low_res_images = low_res_images.to(self.device)
        targets_yolov3["batch_idx"] = targets_yolov3["batch_idx"].to(self.device)
        targets_yolov3["cls"] = targets_yolov3["cls"].to(self.device)
        targets_yolov3["bboxes"] = targets_yolov3["bboxes"].to(self.device)
        combined_ignore_boxes_list = []
        if processed_ignore_list is not None:
            for i, ignore_boxes_for_image in enumerate(processed_ignore_list):
                # ignore_boxes_for_image is expected to be [num_ignore_i, 5] with [class_id, x_c, y_c, w, h] normalized
                if ignore_boxes_for_image.numel() > 0:
                    # Create batch index column (float type to match other tensors)
                    batch_idx_tensor = torch.full((ignore_boxes_for_image.shape[0], 1), i, dtype=torch.float32,
                                                  device=self.device)
                    # Concatenate batch_idx column with the ignore box tensor
                    # Resulting format: [batch_idx, class_id, x_c, y_c, w, h]
                    combined_row = torch.cat([batch_idx_tensor, ignore_boxes_for_image.to(self.device)], dim=1)
                    combined_ignore_boxes_list.append(combined_row)

        # Concatenate all ignore boxes into a single tensor for the batch
        if len(combined_ignore_boxes_list) > 0:
            final_ignored_bboxes_tensor = torch.cat(combined_ignore_boxes_list, dim=0)
            # Add this tensor to the targets dictionary under a specific key.
            # The YOLO loss function (specifically, the assigner within it)
            # must be modified elsewhere to recognize and use this key
            # (e.g., "ignored_bboxes") to mark predictions within these
            # regions as 'ignore' rather than positive or negative.
            targets_yolov3["ignored_bboxes"] = final_ignored_bboxes_tensor
        else:
            # Add an empty tensor if no ignore boxes exist in the batch
            # Ensure the shape matches the expected format [N, 6]
            targets_yolov3["ignored_bboxes"] = torch.empty(0, 6, dtype=torch.float32,
                                                           device=self.device)  # Expected format: [batch_idx, cls, x, y, w, h]
        # --- End of ignore target processing ---

        # --- Training steps ---
        self.optimizer.zero_grad()

        output = self(low_res_images)

        loss_tuple = self.yolov3_wrapper.loss(targets_yolov3, output)
        all_loss_tensors = [loss for group in loss_tuple for loss in group]
        loss = sum(all_loss_tensors)

        loss.backward()

        self.optimizer.step()

        return loss

    def train_epoch(self, train_loader, epoch):
        epoch_loss = 0.0
        train_loader= process_and_return_loaders(train_loader)
        pbar = tqdm(train_loader, total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, tra_batch in enumerate(pbar):
            loss = self.train_step(tra_batch)

            epoch_loss += loss

            if batch_idx % self.config['train']['log_interval'] == 0:
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Batch': f'{batch_idx + 1}/{self.train_batch_nums}'
                })

        avg_loss = epoch_loss / self.train_batch_nums

        print(f"Epoch {epoch} 训练完成，平均 Loss: {avg_loss:.4f}")
        return avg_loss

    def train_model(self, train_loader, val_loader, clean_loader, start_epoch=0, num_epochs=100, checkpoint_dir='./models/DE_NET/checkpoints'):
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
            self.train_epoch(train_loader, epoch)

            # 验证集
            if val_loader:
                self.enhancement.eval()
                self.yolov3.eval()

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

    def evaluate(self, val_loader):
        total_loss = 0.0
        val_loader= process_and_return_loaders(val_loader)
        with torch.no_grad():
            pbar = tqdm(val_loader, total=self.val_batch_nums, desc=f"Val")
            for batch_idx, val_batch in enumerate(pbar):
                low_res_images, targets_yolov3, processed_ignore_list = val_batch
                # print(type(targets_yolov3))
                low_res_images = low_res_images.to(self.device)
                targets_yolov3["batch_idx"] = targets_yolov3["batch_idx"].to(self.device)
                targets_yolov3["cls"] = targets_yolov3["cls"].to(self.device)
                targets_yolov3["bboxes"] = targets_yolov3["bboxes"].to(self.device)
                combined_ignore_boxes_list = []
                if processed_ignore_list is not None:
                    for i, ignore_boxes_for_image in enumerate(processed_ignore_list):
                        # ignore_boxes_for_image is expected to be [num_ignore_i, 5] with [class_id, x_c, y_c, w, h] normalized
                        if ignore_boxes_for_image.numel() > 0:
                            # Create batch index column (float type to match other tensors)
                            batch_idx_tensor = torch.full((ignore_boxes_for_image.shape[0], 1), i, dtype=torch.float32,
                                                          device=self.device)
                            # Concatenate batch_idx column with the ignore box tensor
                            # Resulting format: [batch_idx, class_id, x_c, y_c, w, h]
                            combined_row = torch.cat([batch_idx_tensor, ignore_boxes_for_image.to(self.device)], dim=1)
                            combined_ignore_boxes_list.append(combined_row)

                # Concatenate all ignore boxes into a single tensor for the batch
                if len(combined_ignore_boxes_list) > 0:
                    final_ignored_bboxes_tensor = torch.cat(combined_ignore_boxes_list, dim=0)
                    # Add this tensor to the targets dictionary under a specific key.
                    # The YOLO loss function (specifically, the assigner within it)
                    # must be modified elsewhere to recognize and use this key
                    # (e.g., "ignored_bboxes") to mark predictions within these
                    # regions as 'ignore' rather than positive or negative.
                    targets_yolov3["ignored_bboxes"] = final_ignored_bboxes_tensor
                else:
                    # Add an empty tensor if no ignore boxes exist in the batch
                    # Ensure the shape matches the expected format [N, 6]
                    targets_yolov3["ignored_bboxes"] = torch.empty(0, 6, dtype=torch.float32,
                                                                   device=self.device)  # Expected format: [batch_idx, cls, x, y, w, h]
                # --- End of ignore target processing ---

                # YOLO前向
                output = self(low_res_images)
                loss_tuple = self.yolov3_wrapper.loss(targets_yolov3, output)
                loss = sum([t.sum() for t in loss_tuple])

                total_loss += loss.item()
                if batch_idx % self.config['train']['log_interval'] == 0:
                    pbar.set_postfix({
                        'YOLOv3 Loss': f'{loss:.4f}',
                        'Batch': f'{batch_idx + 1}/{self.val_batch_nums}'
                    })

        avg_loss = total_loss / self.val_batch_nums

        print(
            f"验证集 Avg YOLOv3 Loss: {avg_loss:.4f}")
        return {
            'avg_yolov3_loss': avg_loss
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

    def predict(self, high_res_images):
        self.enhancement.eval()
        self.yolov3.eval()
        high_res_images = high_res_images.to(self.device)
        with torch.no_grad():
            output = self.enhancement(high_res_images)
            results = self.yolov3_wrapper.predict(output)
        return results