# HazeUAVTrack

## 项目简介

**HazeUAVTrack** 是一个专为学术研究设计的综合性平台，旨在为雾天场景下的无人机目标检测与追踪提供一个灵活、模块化的研究和测试环境。该平台基于 **UAV-M 数据集**，支持对不同算法进行高效的对比分析和可视化评估。

系统主要由以下三个核心模块组成：
1.  **图像去雾 (Image Dehazing)**
2.  **目标检测 (Object Detection)**
3.  **目标追踪 (Object Tracking)**

此外，HazeUAVTrack 还集成了合成雾图像生成（通过配置文件选择方法）、批量测试、效果对比和可视化评估等辅助功能。各模块可以独立运行，也可根据任务需求灵活组合进行训练和测试。

通过 HazeUAVTrack，研究人员可以便捷地集成和评估新的去雾、检测或追踪算法，推动雾天无人机感知技术的发展。

## 特性

*   **模块化设计:** 去雾、检测、追踪模块独立实现，易于替换和扩展。
*   **UAV-M 数据集支持:** 专注于 UAV-M 数据集的测试和评估。
*   **灵活的实验配置:** 通过 YAML 文件定义实验流程和算法组合。
*   **集成现有算法:** 已集成多篇论文的去雾和检测算法实现。
*   **全面的评估工具:** 提供去雾效果对比 (PSNR, SSIM, FPS) 和检测/追踪的可视化评估。
*   **合成雾生成:** 支持根据配置生成合成雾图像用于训练或测试。
*   **GPU 内存管理:** `run.py` 脚本包含训练后的内存清理机制。

## 安装指南

### 1. 环境准备

推荐使用 Python 3.12。

### 2. 克隆仓库

``` bash
git clone https://github.com/hczl/HazeUAVTrack.git
cd HazeUAVTrack
```

### 3. 安装依赖

``` bash
pip install -r requirements.txt  
```

### 4. 数据集设置

本项目主要使用 **UAV-M 数据集**。请从以下链接下载数据集，并将其解压到项目的 `data/UAV-M/` 目录下。

**UAV-M 数据集及标签下载链接:**
[百度网盘](https://pan.baidu.com/s/1tVKhiGbr-mg8Z93CFb3R-A?pwd=vti5)

**UAV-M 雾浓度0.5版本下载链接:**
[百度网盘](https://pan.baidu.com/s/1qmrnfyxkmwOS2PueUVZyGQ?pwd=vti5)

**UAV-M 雾浓度0.75版本下载链接:**
[百度网盘](https://pan.baidu.com/s/1WgE9XaYaio_DyWHT1A6lIg?pwd=vti5)

请确保解压后的数据集文件位于 `HazeUAVTrack/data/UAV-M/` 路径下。

## 项目结构
```
HazeUAVTrack/
├── configs/          # YAML配置文件目录 (已包含现有配置，如 AD_NET.yaml, DENet.yaml 等)
├── data/             # 数据集目录
│   └── UAV-M/        # UAV-M 数据集存放位置 (请将下载的数据集解压到此处)
├── models/           # 模型代码目录 (存放去雾、检测、追踪等模型的实现)
│   ├── dehaze/       # 去雾模型代码，例如 aodnet.py, falcon.py, ffanet.py, your_model_file.py
│   ├── detector/     # 检测模型代码，例如 denet.py, iayolo.py, your_model_file.py
│   └── tracker/      # 追踪器代码 (如果实现为独立模块)
├── result/           # 实验结果输出目录
│   ├── dehaze/       # 去雾测试结果文件
│   └── video/        # 视频可视化结果文件
├── app.py               # 可视化UI应用的主文件
├── utils/            # 工具函数目录 (create_model, create_data, load_config等)
├── dehaze_test.py    # 去雾效果对比测试脚本
├── run.py            # 运行实验主脚本
├── requirements.txt  # Python依赖列表 (如果手动创建)
└── README.md         # 本文件
```

## 如何运行

### 1. 配置实验

实验的配置通过 `configs/` 目录下的 YAML 文件定义。每个 YAML 文件定义了一组实验参数，包括使用的去雾、检测、追踪方法等。

现有配置均对应相应论文方法名，详细参数注释请参考`configs/stripped_comments.json`文件

### 2. 运行训练和测试 (run.py)
run.py 脚本负责加载配置、创建数据加载器和模型，执行训练过程。它设计用于批量运行多个实验配置。

脚本会读取 run.py 中定义的 YAML 文件列表，依次运行每个实验。

#### run.py 示例片段
``` python
yaml = ['AD_NET', 'FFA_NET_YOLO'] # 你可以在这里添加更多你想运行的配置文件的名字 (不带.yaml后缀)
for i in yaml:
    # ... (内存清理等代码)
    run_experiment(f"configs/{i}.yaml")
    ... (最终内存清理代码)
```

脚本会按照列表顺序运行指定的 YAML 配置。在每个实验完成后，脚本会尝试进行 GPU 内存清理，以确保连续运行多个实验时不会耗尽显存。

训练后的模型通常会保存在项目指定的输出路径（具体路径可能在配置文件中定义或有默认设置）。

### 3. 测试去雾效果 (dehaze_test.py)
dehaze_test.py 脚本用于独立测试不同去雾方法的性能，并计算 PSNR, SSIM, FPS 等指标。

要运行去雾测试：

``` 
python dehaze_test.py
```

测试结果将保存在 result/dehaze/ 目录下。

### 4. 可视化结果 (app/app.py)
app.py 脚本包含一个用于可视化检测和追踪结果的 UI 应用。你可以选择一个 YAML 配置文件和一个视频文件夹，应用会加载对应的模型和视频，并显示带检测/追踪框的可视化结果。

要启动可视化 UI：

``` 
python app.py
```

# 添加新的方法
本项目采用模块化设计，方便集成新的去雾、检测或追踪算法。

## 1. 实现新的模型代码
在 `models/dehaze/`, `models/detector/`, 或 `models/tracker/` 目录下创建你的模型实现文件（例如 `your_new_dehaze_model.py`）。

请确保你的模型实现符合以下接口要求：

### a. 添加新的去雾模块

在 `models/dehaze/` 目录下创建你的模型文件。你的模型类需要包含一个`forward_loss`方法，用于训练，并且需要类名与文件名一样，例如：

``` python
# 示例：models/dehaze/your_new_dehaze_model.py
import torch
import torch.nn as nn

class your_new_dehaze_model(nn.Module):
    def __init__(self, config): # config 是从yaml加载的参数字典
        super().__init__()
        # 构建你的模型层
        pass

    def forward(self, haze_img):
        # 前向推理，输入bchw，输出bchw去雾图像
        dehazed_img = ...
        return dehazed_img

    def forward_loss(self, haze_img, clean_img):
        """
        计算去雾模型的损失。
        Args:
            haze_img (torch.Tensor): 输入的雾天图像，形状 (B, C, H, W)。
            clean_img (torch.Tensor): 对应的清晰图像，形状 (B, C, H, W)。

        Returns:
            dict: 包含损失项的字典。必须包含 'total_loss' 键，其值是用于backward的loss tensor。
        """
        dehazed_img = self.forward(haze_img)
        # 计算损失，例如 MSE Loss
        loss = nn.functional.mse_loss(dehazed_img, clean_img)

        return {
            'total_loss': loss,
            # 你可以添加其他你想记录的损失项，例如 'mse_loss': loss
        }
```

### b. 添加新的目标检测模块
在 models/detector/ 目录下创建你的模型文件。你的模型类需要包含一个 forward_loss 方法用于训练和一个 predict 方法用于推理，且需要类名与文件名一样，例如:
``` python
# 示例: models/detector/your_new_detector.py
import torch
import torch.nn as nn

class your_new_detector(nn.Module):
    def __init__(self, config): # config 是从yaml加载的参数字典
        super().__init__()
        # 构建你的模型层
        pass

    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):
        """
        计算检测模型的损失。
        Args:
            dehaze_imgs (torch.Tensor): 输入的图像 (可以是去雾后的)，形状 (B, C, H, W)。
            targets (list of torch.Tensor): 包含batch个list，每个list对应一张图像的ground truth targets。
                                            每个target tensor的形状是 (N, 9)，列对应:
                                            <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
                                            注意：这里的bbox是xywh格式
            ignore_list (list of torch.Tensor, optional): 类似于targets，用于标记在计算损失时应忽略的目标。

        Returns:
            dict: 包含损失项的字典。必须包含 'total_loss' 键，其值是用于backward的loss tensor。
        """
        # 在这里计算模型的损失，例如分类损失、回归损失等
        # 你需要根据targets计算损失
        # 请注意，targets中的bbox是xywh格式，可能需要转换为xyxy格式或其他格式进行计算
        loss = ... # 计算你的总损失

        return {
            'total_loss': loss,
            # 你可以添加其他你想记录的损失项，例如 'cls_loss': cls_loss, 'reg_loss': reg_loss
        }

    def predict(self, image):
        """
        对单张图像进行推理。
        Args:
            image (torch.Tensor): 输入的单张图像，形状 (C, H, W) 或 (1, C, H, W)。

        Returns:
            torch.Tensor: 检测结果，形状 (N, 5)，每行为 [x1, y1, x2, y2, confidence]。
                          坐标是xyxy格式。
        """
        # 在这里执行单张图像的推理，返回检测框和置信度
        detection_results = ... # 形状 (N, 5)，每行 [x1, y1, x2, y2, confidence]
        return detection_results
```

### c. 添加新的目标追踪模块
如果你的追踪器是作为一个独立模块实现的（例如，使用检测器的输出进行追踪），在 models/tracker/ 目录下创建你的文件（例如 your_new_tracker.py）。你的追踪器类需要包含一个初始化方法和一个 update 方法，且需要类名与文件名一样。
``` python
# 示例: models/tracker/your_new_tracker.py
import torch
import torch.nn as nn
import numpy as np # 如果你的追踪器需要 NumPy 格式的图像

class YourNewTracker(nn.Module): # 可以继承nn.Module，如果需要保存状态或参数
    def __init__(self, cfg): # config 是从yaml加载的参数字典
        super().__init__()
        self.cfg = cfg
        # 根据 cfg 初始化你的追踪器逻辑

    def update(self, detections, image_tensor, frame_id=None):
        """
        根据当前帧的检测结果和图像更新追踪状态。
        Args:
            detections (list or np.ndarray): 当前帧的检测结果。格式取决于你的检测器输出和追踪器输入（例如，N个目标，每个目标 [x1, y1, x2, y2, confidence, class_id] 或 [x1, y1, x2, y2, confidence]）。
                                             请根据你的实际集成情况调整这里的描述。
            image_tensor (torch.Tensor): 当前帧的图像，形状通常是 (1, C, H, W) 或 (B, C, H, W)。
            frame_id (int, optional): 当前帧的ID。如果提供，使用此ID；否则内部递增。

        Returns:
            list or np.ndarray: 更新后的追踪结果。格式取决于你的追踪器输出（例如，M个追踪目标，每个目标 [x1, y1, x2, y2, track_id, ...]）。
                                请根据你的实际集成情况调整这里的描述。BoxMOT 返回的是 NumPy 数组。
        """
        self.frame_id += 1 if frame_id is None else frame_id

        # 许多追踪库需要 NumPy 格式的图像，且通道顺序可能是 HWC
        # 根据需要进行图像格式转换
        # img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # 示例转换 BCHW -> HWC NumPy

        # 调用你的内部追踪器或实现追踪逻辑
        # 例如：
        # tracks = self.internal_tracker.process_frame(detections, img_np, self.frame_id)

        # 示例文本，替换为你的实际追踪逻辑和返回值
        print(f"Processing frame {self.frame_id} with {len(detections)} detections...")
        tracks = [] # 你的追踪结果列表或数组

        return tracks
```
## 2. 创建 YAML 配置文件

在 configs/ 目录下创建一个新的 YAML 文件（例如 YourNewMethod.yaml），在其中指定使用你新实现的方法，参数参考configs/stripped_comments.json文件。

## 3. 运行新方法
将你的新 YAML 文件名（不带 .yaml 后缀）添加到 run.py 脚本中的 yaml 列表中，然后运行 python run.py 即可开始使用你的新方法进行实验。

## 已集成的论文方法

本项目已集成以下论文的算法实现：

### 图像去雾 (Image Dehazing)

| Method Name | Paper |
| --- | --- |
| AOD-Net | Li J, Brown M S. AOD-Net: All-in-One Dehazing Network[C]//2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017: 2951-2959. |
| FALCON | Kim H, Lee J, Park S. FALCON: Frequency Adjoint Link with Continuous Density Mask for Real-Time Dehazing[J]. IEEE Transactions on Image Processing, 2024, 33: 1234-1246. |
| FFA-Net | X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia, "FFA-Net: Feature fusion attention network for single image dehazing," Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 1-10. |

### 目标检测 (Object Detection)

| Method Name | Paper |
| --- | --- |
| DENet | Qin Q, Chang K, Huang M, et al. DENet: Detection-driven Enhancement Network for Object Detection Under Adverse Weather Conditions[M]//Wang L, Gall J, Chin T J, et al. Computer Vision – ACCV 2022: Vol. 13843. Cham: Springer Nature Switzerland, 2023: 491-507. |
| Image-Adaptive YOLO | Liu W, Ren G, Yu R, et al. Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions[A]. arXiv, 2022. |

## 实验结果与输出

*   **模型保存:** 训练完成后的模型权重文件会根据配置保存到指定目录。
*   **去雾测试结果:** 运行 `dehaze_test.py` 生成的 PSNR, SSIM, FPS 等指标数据和图表将保存在 `result/dehaze/` 目录下。
*   **视频可视化:** 使用 `app/app.py` 生成的带检测/追踪框的视频文件将保存在 `result/video/` 目录下。

## 贡献与问题反馈

欢迎对本项目做出贡献或提交问题反馈！

*   **报告 Bug:** 如果你发现了 Bug，请在 GitHub Issues 页面提交，并尽可能提供详细的复现步骤和环境信息。
*   **提交功能请求:** 如果你有新的功能想法，也欢迎在 GitHub Issues 页面提出。
*   **贡献代码:**
    1.  Fork 本仓库。
    2.  创建你的功能分支 (`git checkout -b feature/YourFeature`).
    3.  提交你的修改 (`git commit -am 'Add some feature'`).
    4.  推送到分支 (`git push origin feature/YourFeature`).
    5.  提交一个 Pull Request。

## 许可协议

本项目根据 **GNU Affero General Public License v3.0 (AGPL-3.0)** 发布。由于本项目集成了使用 AGPL-3.0 许可证的外部库 (如 BoxMOT 和 Ultralytics YOLOv3)，根据 AGPL-3.0 的条款，本项目也必须采用 AGPL-3.0 许可证。这意味着如果你分发本项目（包括通过网络提供服务），你必须提供本项目的完整源代码，并在 AGPL-3.0 的条款下可用。请参阅仓库中的 `LICENSE` 文件获取完整的许可协议文本。

## 致谢

感谢所有为本项目做出贡献的开发者，以及 UAV-M 数据集的提供者。特别感谢 [BoxMOT](https://github.com/mikel-brostrom/boxmot) 和 [Ultralytics YOLOv3](https://github.com/ultralytics/yolov3) 项目的开发者，他们的优秀工作为本项目提供了关键支持。
