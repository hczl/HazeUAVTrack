import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import time
import math
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm # Using tqdm for console output in the worker thread

try:
    # 尝试从 utils 文件夹导入配置加载和模型创建函数
    from utils.config import load_config
    from utils.create import create_model
except ImportError as e:
    # 如果导入失败，打印错误并提示用户检查路径
    print(f"Error importing utility functions: {e}")
    print("Please ensure 'utils' folder with config.py and create.py is in your project path.")
    # 如果工具函数是必需的，抛出错误或进行适当处理
    raise

# 设置 PyTorch Hub 的缓存目录，避免每次下载
os.environ['TORCH_HOME'] = './.torch' # 示例：设置为当前目录下的 .torch 文件夹

# ---- 用于线程通信的全局/共享变量 ----
# 用于存储工作线程处理好的帧数据 (Tkinter PhotoImage 对象和状态文本)
current_frame_data = None # 初始为 None
processing_running = False # 标志，指示工作线程是否正在运行
stop_event = threading.Event() # 事件对象，用于向工作线程发送停止信号

# ---- 模型和处理相关的全局变量 (由工作线程加载) ----
model = None # 模型实例
device = None # 设备 (CPU 或 GPU)
cfg = None # 配置字典
# 图像预处理变换：转换为 PyTorch Tensor
transform = transforms.Compose([transforms.ToTensor()])
image_files = [] # 视频帧文件列表
current_frame_index = -1 # 当前处理的帧索引，-1 表示未开始或已完成
max_size = 1024 # 默认最大图像尺寸 (用于模型输入)

# ---- 图像处理和检测函数 (改编自您的脚本) ----
def preprocess_image(image_pil, max_size):
    """
    预处理 PIL 图像用于模型输入：转换为 Tensor，调整大小，添加批次维度。

    Args:
        image_pil (PIL.Image.Image): 输入的 PIL 图像。
        max_size (int): 模型输入的最大边长。

    Returns:
        tuple: 包含 (input_tensor, original_dimensions, resized_dimensions)。
               input_tensor (Tensor): 处理后用于模型输入的张量 [1, C, H', W']。
               original_dimensions (tuple): 原始图像尺寸 (orig_w, orig_h)。
               resized_dimensions (tuple): 调整后的模型输入尺寸 (new_w, new_h)。
    """
    orig_w, orig_h = image_pil.size # 原始尺寸
    image_tensor = transform(image_pil) # 转换为 Tensor [C, H, W]

    # 计算调整大小的比例因子，确保最长边不超过 max_size
    r = min(1.0, max_size / float(max(orig_w, orig_h)))
    # 计算新的尺寸，确保是 32 的倍数 (模型常见要求)
    new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
    new_w = max(32, int(math.floor(orig_w * r / 32) * 32))

    # 使用 F.resize 调整尺寸。注意 F.resize 接收 (height, width) 元组。
    # 这里的 resize 会将图像拉伸/压缩到新的 (new_h, new_w) 尺寸，不一定保持纵横比。
    # 如果需要保持纵横比并填充，需要更复杂的逻辑。当前代码是简单 resize。
    image_resized_tensor = F.resize(image_tensor, (new_h, new_w))
    # 假设模型期望 float Tensor，值在 [0, 1] 范围
    input_tensor = image_resized_tensor.unsqueeze(0).to(device) # 添加批次维度并发送到设备

    # 返回输入张量、原始尺寸和调整后尺寸
    return input_tensor, (orig_w, orig_h), (new_w, new_h)

def scale_boxes_to_original(boxes, orig_dims, new_dims):
    """
    将调整大小后的图像坐标系中的预测框缩放到原始图像坐标系。

    Args:
        boxes (list or Tensor): 预测的边界框列表或 Tensor，坐标相对于调整后的尺寸。
                                格式应为 [x1, y1, x2, y2, ...]
        orig_dims (tuple): 原始图像尺寸 (orig_w, orig_h)。
        new_dims (tuple): 调整后图像尺寸 (new_w, new_h)。

    Returns:
        list: 缩放回原始尺寸的边界框列表。
    """
    orig_w, orig_h = orig_dims
    new_w, new_h = new_dims

    # 根据简单的拉伸/压缩 resize 计算缩放因子
    # 确保 new_w 或 new_h 不为零，避免除以零 (如果 max_size > 0 通常不会发生)
    scale_w = orig_w / new_w if new_w > 0 else 1.0
    scale_h = orig_h / new_h if new_h > 0 else 1.0

    scaled_boxes = []
    # 遍历每个预测框
    for box in boxes:
        # 确保框数据至少包含 x1, y1, x2, y2
        if len(box) < 4:
            continue # 跳过格式不正确的框

        # 如果输入是 Tensor，转换为 list 或 numpy 方便处理
        if isinstance(box, torch.Tensor):
             box_list = box.tolist()
        elif isinstance(box, np.ndarray):
             box_list = box.tolist()
        else:
             box_list = box # 已经是 list 或其他可迭代格式

        x1, y1, x2, y2 = box_list[:4]
        # 应用缩放
        x1 *= scale_w
        y1 *= scale_h
        x2 *= scale_w
        y2 *= scale_h
        # 将缩放后的坐标与其他信息 (如置信度、类别等) 一起添加到结果列表
        scaled_boxes.append([x1, y1, x2, y2] + box_list[4:])

    return scaled_boxes


def draw_boxes(image_np_original_size, detections):
    """
    在原始尺寸的 OpenCV 图像 (BGR 格式) 上绘制边界框和标签。

    Args:
        image_np_original_size (np.ndarray): 原始尺寸的 OpenCV 图像 (BGR, uint8)。
        detections (list): 检测结果列表，坐标必须相对于原始尺寸。
                           格式应为 [x1, y1, x2, y2, conf, ...]。
                           假设这些检测结果已经由模型根据置信度和 IoU 进行了过滤。

    Returns:
        np.ndarray: 绘制了边界框的 OpenCV 图像 (BGR 格式)。
    """
    # image_np_original_size: 原始尺寸的 OpenCV 图像，BGR 格式，uint8
    # detections: 检测结果列表，坐标必须是原始尺寸的坐标。
    # 假设 detections 已经是模型过滤后的结果。
    img_to_draw = image_np_original_size.copy() # 在图像副本上绘制，避免修改原图

    # 这里不再根据置信度进行过滤，假设模型输出已过滤
    filtered_detections = detections # 使用提供的所有检测结果
    for det in filtered_detections:
        # 确保检测结果至少包含 5 个元素 (x1, y1, x2, y2, conf)
        if len(det) < 5:
            continue # 跳过格式不正确的检测结果

        x1, y1, x2, y2, conf = det[:5] # 只使用前 5 个元素
        # 确保坐标是整数，用于 OpenCV 绘图
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制矩形框 (OpenCV 使用 BGR 颜色格式)
        # 绿色 (0, 255, 0)，线宽 2
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加文本标注 (置信度)
        text_x, text_y = x1, y1 - 10 # 文本位置稍高于框的顶部
        text_y = max(text_y, 15) # 确保文本不会超出图像顶部边界

        label = f"{conf:.2f}" # 格式化置信度为字符串

        # 绘制文本 (OpenCV 使用 BGR 颜色格式)
        # 黄色 (0, 255, 255)，字体缩放 0.5，线宽 2
        cv2.putText(img_to_draw, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img_to_draw


# ---- 工作线程函数 ----
def process_video_frames(yaml_path, image_folder, conf_var, iou_var, status_var, video_label_size):
    """
    此函数在单独的线程中运行，执行视频处理任务。
    它更新 UI 线程读取的共享变量。
    它还会根据滑动条的值设置模型的阈值。

    Args:
        yaml_path (str): 模型配置 YAML 文件路径。
        image_folder (str): 包含视频帧图片的文件夹路径。
        conf_var (tk.DoubleVar): 共享变量，存储置信度阈值。
        iou_var (tk.DoubleVar): 共享变量，存储 IoU 阈值。
        status_var (tk.StringVar): 共享变量，存储状态文本。
        video_label_size (tuple): UI 中视频显示区域的尺寸 (宽, 高)，用于调整显示图片大小。
    """
    global model, device, cfg, image_files, current_frame_index, processing_running, current_frame_data, max_size

    processing_running = True # 设置运行标志
    current_frame_index = -1 # 重置帧索引

    # --- 加载配置和模型 ---
    try:
        status_var.set("Status: Loading config...") # 更新 UI 状态
        cfg = load_config(yaml_path) # 加载配置
        max_size = cfg.get('max_size', 1024) # 从配置中获取 max_size，默认为 1024

        status_var.set("Status: Creating model...") # 更新 UI 状态
        model = create_model(cfg) # 创建模型实例

        status_var.set("Status: Loading model weights...") # 更新 UI 状态
        model.load_model() # 加载模型权重

        # 设置设备，优先使用 CUDA
        device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
        model.to(device) # 将模型发送到设备
        model.eval() # 设置模型为评估模式 (关闭 dropout 等)

        # --- 从配置设置模型的初始阈值 ---
        initial_conf = cfg.get('detector_conf_thresh', 0.25) # 从配置获取初始置信度阈值
        initial_iou = cfg.get('detector_iou_thresh', 0.45) # 从配置获取初始 IoU 阈值
        try:
            # 尝试直接设置模型的属性
            if hasattr(model, 'conf_thresh'):
                model.conf_thresh = initial_conf
                conf_var.set(initial_conf) # 也将 UI 滑动条的值设置为初始配置值
            else:
                 print("Warning: Model object does not have a 'conf_thresh' attribute.")
                 print("Confidence slider will control display filtering only (if draw_boxes was filtering).")


            if hasattr(model, 'iou_thresh'):
                model.iou_thresh = initial_iou
                iou_var.set(initial_iou) # 也将 UI 滑动条的值设置为初始配置值
            else:
                 print("Warning: Model object does not have an 'iou_thresh' attribute.")
                 print("IoU slider value will only be displayed.")

            status_var.set(f"Status: Model loaded on {device}. Initial thresholds: Conf={initial_conf:.2f}, IoU={initial_iou:.2f}. Warming up...") # 更新 UI 状态

        except Exception as e:
             print(f"Error setting initial model thresholds: {e}")
             status_var.set(f"Status: Model loaded on {device}. Error setting initial thresholds ({e}). Warming up...")


        # --- 获取图片文件列表 ---
        image_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder) # 遍历文件夹内容
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) # 筛选图片文件
        ])

        if not image_files:
            status_var.set(f"Error: No image files found in {image_folder}") # 更新 UI 状态
            processing_running = False # 设置运行标志为 False
            # 如果加载失败，清理模型
            if model is not None:
                del model
                model = None
            if torch.cuda.is_available():
                 torch.cuda.empty_cache() # 清理 CUDA 缓存
            return # 退出函数

    except Exception as e:
        status_var.set(f"Error during model loading: {e}") # 更新 UI 状态
        model = None # 确保模型为 None 如果加载失败
        processing_running = False # 设置运行标志为 False
        return # 退出函数

    status_var.set("Status: Model ready. Starting inference...") # 更新 UI 状态
    time.sleep(1) # 暂停一秒，给 UI 时间更新状态

    start_time = time.time() # 记录开始时间
    frame_count = 0 # 初始化帧计数器

    # --- 主处理循环 ---
    # 使用 tqdm 显示控制台进度条
    for i, image_path in enumerate(tqdm(image_files, desc="Processing Frames")):
        if stop_event.is_set(): # 检查停止事件是否被设置
            status_var.set("Status: Processing stopped.") # 更新 UI 状态
            break # 如果停止事件被设置，则跳出循环，停止处理

        current_frame_index = i # 更新当前帧索引
        frame_start_time = time.time() # 记录当前帧开始处理时间

        # --- 从 UI 滑动条获取当前的阈值 ---
        # 在预测之前读取共享的 DoubleVar 对象的值
        current_conf_thresh = conf_var.get()
        current_iou_thresh = iou_var.get()

        # --- 为当前帧设置模型的阈值 ---
        # 这里更新模型的内部状态，使预测使用最新的阈值
        try:
            if model is not None: # 确保模型已成功加载
                if hasattr(model, 'conf_thresh'):
                    model.conf_thresh = current_conf_thresh
                if hasattr(model, 'iou_thresh'):
                    model.iou_thresh = current_iou_thresh
        except Exception as e:
            # 这个警告可能会很频繁，可以选择记录日志或用其他方式处理
            # print(f"Warning: Could not set model thresholds for frame {i}: {e}")
            pass # 忽略频繁警告


        try:
            # 加载原始图片，用于后续绘制
            image_pil_original = Image.open(image_path).convert("RGB") # 使用 Pillow 打开并转为 RGB
            orig_w, orig_h = image_pil_original.size # 获取原始尺寸
            # 将原始 PIL 图像转换为 OpenCV BGR 格式，用于绘制
            image_np_original_bgr = cv2.cvtColor(np.array(image_pil_original), cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Worker: Skipping file {image_path}: Cannot open or process image ({e})")
            continue # 跳过当前文件，处理下一张图片

        # 1. 预处理图像 (用于模型输入 - 这需要调整大小)
        input_tensor, orig_dims, new_dims = preprocess_image(image_pil_original, max_size)

        # 2. 去雾和预测
        with torch.no_grad(): # 在推理时禁用梯度计算
            # 获取去雾后的图像 (这是模型输出的调整后尺寸的图像)
            # 我们会将其调整回原始尺寸用于绘制。
            if model is not None: # 确保模型可用
                # 假设模型有 dehaze 方法，返回去雾后的 Tensor
                # 如果模型没有单独的 dehaze 方法，或者预测函数返回了去雾图像，需要调整这里
                # 注意：原始的 DRIFT_NET 代码没有 dehaze 方法。这里假设模型返回的某个输出是去雾图像。
                # 或者 dehaze 只是一个占位符。需要根据实际模型实现调整。
                # 如果模型只输出预测框，那么 dehazed_tensor 部分可能需要移除或替换。
                # 为了演示，假设 model.dehaze(input_tensor) 返回去雾图像 Tensor。
                # 如果模型没有 dehaze 方法，可以跳过这步，直接用原始图像进行绘制。
                try:
                    # 假设模型有 dehaze 方法
                    dehazed_tensor = model.dehaze(input_tensor) # [1, C, H', W']
                    # 将去雾后的 Tensor 转换为 OpenCV 格式 (NumPy BGR uint8) - 仍然是调整后的尺寸
                    # Tensor 格式通常是 [C, H, W]，需要 permute 到 [H, W, C]
                    dehazed_np_resized = (dehazed_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    dehazed_np_resized = cv2.cvtColor(dehazed_np_resized, cv2.COLOR_RGB2BGR)
                except AttributeError:
                    # 如果模型没有 dehaze 方法，使用原始图像的 resized 版本
                    print("Warning: Model does not have a 'dehaze' method. Using resized original image.")
                    # 将原始图像 resize 到模型输入尺寸作为“去雾”图像
                    dehazed_np_resized = cv2.resize(image_np_original_bgr, (new_dims[0], new_dims[1]), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    print(f"Error during dehazing: {e}. Using resized original image.")
                    # 如果 dehaze 过程中发生其他错误，也使用原始图像的 resized 版本
                    dehazed_np_resized = cv2.resize(image_np_original_bgr, (new_dims[0], new_dims[1]), interpolation=cv2.INTER_LINEAR)


                # 获取预测结果。模型应该使用上面设置的内部 conf_thresh 和 iou_thresh 来过滤预测。
                # model.predict 应返回相对于 new_dims 的预测框列表或 Tensor。
                predictions = model.predict(input_tensor) # 这些坐标是相对于 new_dims 的

            else: # 模型加载失败，使用占位符
                 # 如果模型加载失败，使用原始图像的 resized 版本作为显示图像
                 dehazed_np_resized = cv2.resize(image_np_original_bgr, (new_dims[0], new_dims[1]), interpolation=cv2.INTER_LINEAR)
                 predictions = [] # 没有预测结果如果模型失败


        # 3. 将预测框从调整后尺寸坐标缩放回原始尺寸坐标
        # 这些预测结果应该已经由模型的内部阈值过滤过了
        scaled_predictions = scale_boxes_to_original(predictions, orig_dims, new_dims)

        # 4. 将去雾后的图像调整回原始尺寸，用于绘制和显示
        dehazed_np_original_size = cv2.resize(dehazed_np_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


        # 5. 在原始尺寸的去雾图像上绘制结果。
        # 这里不做置信度过滤，假设模型已经完成了。
        img_to_draw = draw_boxes(dehazed_np_original_size, scaled_predictions)

        # 6. 计算 FPS 并准备状态文本
        frame_end_time = time.time() # 记录当前帧结束处理时间
        frame_time = frame_end_time - frame_start_time # 计算单帧处理时间
        frame_count += 1 # 增加帧计数
        elapsed_time = time.time() - start_time # 计算总处理时间
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0 # 计算 FPS

        # 构建状态文本字符串
        status_text = f"FPS: {fps:.2f} | Conf: {current_conf_thresh:.2f} | IoU: {current_iou_thresh:.2f} | Frame: {i+1}/{len(image_files)}"

        # 7. 准备最终绘制了框的图像 (原始尺寸) 用于 Tkinter 显示
        # 将 BGR 格式的 OpenCV 图像转换为 RGB 格式的 PIL 图像
        img_rgb_original_size = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
        img_pil_original_size = Image.fromarray(img_rgb_original_size)

        # 根据 UI 视频显示区域的大小调整图像大小，保持纵横比
        if video_label_size and video_label_size[0] > 0 and video_label_size[1] > 0:
             img_pil_display = img_pil_original_size.copy() # 在副本上操作
             # 使用 thumbnail 方法保持纵横比并调整大小
             img_pil_display.thumbnail(video_label_size, Image.Resampling.LANCZOS)
        else:
             # 如果标签尺寸无效，使用原始尺寸 (可能导致显示问题)
             img_pil_display = img_pil_original_size

        # 将 PIL 图像转换为 Tkinter PhotoImage 格式
        img_tk = ImageTk.PhotoImage(img_pil_display)

        # 更新共享变量。
        # 这是一个简单的线程间通信机制：工作线程准备好数据后，将其赋给全局变量。
        global current_frame_data
        current_frame_data = (img_tk, status_text)

        # 可以添加一个小的暂停，控制帧率或减少 CPU 负载
        # time.sleep(0.005) # 根据需要调整

    # --- 处理完成 ---
    status_var.set("Status: Processing complete.") # 更新 UI 状态
    processing_running = False # 设置运行标志为 False
    current_frame_index = len(image_files) # 设置帧索引为总帧数，表示完成

    # 清理全局模型和 CUDA 缓存
    if model is not None:
        del model
        model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Tkinter UI 类 ----
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Video Inference UI") # 设置窗口标题
        self.geometry("800x700") # 设置窗口初始尺寸

        # 路径变量，带有默认值
        # !!! 在这里设置您想要的默认路径 !!!
        self.yaml_path_var = tk.StringVar(value="configs/DRIFT_NET.yaml") # 示例默认 YAML 路径
        # 示例默认视频帧文件夹路径
        self.video_folder_var = tk.StringVar(value="data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog_050/M1005")
        # !!! ---------------------------------- !!!


        # 滑动条变量 (DoubleVar 用于存储浮点数)
        self.conf_var = tk.DoubleVar(value=0.25) # 默认置信度 (将从配置更新)
        self.iou_var = tk.DoubleVar(value=0.45)  # 默认 IoU (将从配置更新)

        # 状态文本变量 (StringVar 用于存储字符串)
        self.status_var = tk.StringVar(value="Status: Waiting for input...")

        # 处理线程句柄
        self.processing_thread = None

        # 保持对当前 PhotoImage 的引用，防止垃圾回收
        self.current_tk_image = None

        # UI 布局设置
        self._setup_input_ui() # 设置输入界面
        self._setup_video_ui() # 设置视频显示界面

        # 初始时隐藏视频显示界面
        self.video_frame.pack_forget()

        # 处理窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 调度周期性 UI 更新
        self._update_frame_display_scheduled = None
        # 在处理开始前，我们不启动更新循环

        # 设置初始按钮状态
        self._set_input_ui_state(True) # 启用输入界面控件
        self._set_video_ui_state(False) # 初始隐藏视频界面按钮 (实际是禁用，因为框架未 pack)


    def _setup_input_ui(self):
        """设置输入参数选择界面"""
        self.input_frame = ttk.Frame(self, padding="10")
        self.input_frame.pack(fill=tk.BOTH, expand=True) # 填充并扩展以适应窗口

        # YAML 路径输入
        ttk.Label(self.input_frame, text="Model YAML:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.yaml_entry = ttk.Entry(self.input_frame, textvariable=self.yaml_path_var, width=50)
        self.yaml_entry.grid(row=0, column=1, pady=5, padx=5)
        self.yaml_browse_btn = ttk.Button(self.input_frame, text="Browse", command=self._browse_yaml)
        self.yaml_browse_btn.grid(row=0, column=2, pady=5, padx=5)

        # 视频文件夹输入
        ttk.Label(self.input_frame, text="Video Folder:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.video_folder_entry = ttk.Entry(self.input_frame, textvariable=self.video_folder_var, width=50)
        self.video_folder_entry.grid(row=1, column=1, pady=5, padx=5)
        self.video_folder_browse_btn = ttk.Button(self.input_frame, text="Browse", command=self._browse_folder)
        self.video_folder_browse_btn.grid(row=1, column=2, pady=5, padx=5)

        # 开始按钮
        self.start_button = ttk.Button(self.input_frame, text="Start Processing", command=self._start_processing)
        self.start_button.grid(row=2, column=0, columnspan=3, pady=20)

        # 配置网格布局，使中间列 (entry) 扩展
        self.input_frame.columnconfigure(1, weight=1)

    def _setup_video_ui(self):
        """设置视频显示和控制界面"""
        self.video_frame = ttk.Frame(self, padding="10")
        # 注意: 这个框架初始时没有 pack

        # 状态标签 (显示 FPS, 阈值, 帧数等)
        self.status_label = ttk.Label(self.video_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)

        # 视频显示区域 (Label 用于显示图片)
        self.video_label = ttk.Label(self.video_frame, text="Model warming up...", anchor="center")
        self.video_label.pack(pady=10, fill=tk.BOTH, expand=True) # 填充并扩展，适应父容器
        self.video_label.config(compound=tk.CENTER) # 使文本和图片居中显示

        # 滑动条框架
        sliders_frame = ttk.Frame(self.video_frame)
        sliders_frame.pack(pady=10)

        # 置信度滑动条
        ttk.Label(sliders_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5)
        self.conf_slider = ttk.Scale(sliders_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.conf_var, length=300)
        self.conf_slider.grid(row=0, column=1, padx=5)
        # 显示置信度值的标签
        self.conf_value_label = ttk.Label(sliders_frame, textvariable=self.conf_var, width=5)
        self.conf_value_label.grid(row=0, column=2, padx=5)


        # IoU 滑动条
        ttk.Label(sliders_frame, text="IoU Threshold:").grid(row=1, column=0, padx=5)
        self.iou_slider = ttk.Scale(sliders_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.iou_var, length=300)
        self.iou_slider.grid(row=1, column=1, padx=5)
        # 显示 IoU 值的标签
        self.iou_value_label = ttk.Label(sliders_frame, textvariable=self.iou_var, width=5)
        self.iou_value_label.grid(row=1, column=2, padx=5)

        # 停止并重新选择按钮
        self.stop_button = ttk.Button(self.video_frame, text="Stop and Re-select Video", command=self._stop_and_reselect)
        self.stop_button.pack(pady=10)

    def _set_input_ui_state(self, enabled):
        """启用或禁用输入界面的控件"""
        state = "normal" if enabled else "disabled"
        self.yaml_entry.config(state=state)
        self.yaml_browse_btn.config(state=state)
        self.video_folder_entry.config(state=state)
        self.video_folder_browse_btn.config(state=state)
        self.start_button.config(state=state)

    def _set_video_ui_state(self, enabled):
        """启用或禁用视频界面的控件"""
        state = "normal" if enabled else "disabled"
        # status_label 和 video_label 没有 state 属性
        self.conf_slider.config(state=state)
        self.iou_slider.config(state=state)
        # 值标签没有 state 属性
        self.stop_button.config(state=state)


    def _browse_yaml(self):
        """打开文件对话框选择 YAML 文件"""
        filename = filedialog.askopenfilename(
            title="Select Model YAML File", # 对话框标题
            filetypes=(("YAML files", "*.yaml"), ("All files", "*.*")) # 文件类型过滤器
        )
        if filename:
            self.yaml_path_var.set(filename) # 更新 YAML 路径变量

    def _browse_folder(self):
        """打开文件夹对话框选择视频帧文件夹"""
        foldername = filedialog.askdirectory(
            title="Select Video Frames Folder" # 对话框标题
        )
        if foldername:
            self.video_folder_var.set(foldername) # 更新视频文件夹变量

    def _start_processing(self):
        """开始处理按钮的回调函数"""
        yaml_path = self.yaml_path_var.get() # 获取 YAML 路径
        video_folder = self.video_folder_var.get() # 获取视频文件夹路径

        # 验证路径是否存在
        if not os.path.isfile(yaml_path):
            messagebox.showerror("Error", f"YAML file not found: {yaml_path}")
            return
        if not os.path.isdir(video_folder):
            messagebox.showerror("Error", f"Video folder not found: {video_folder}")
            return

        # 禁用输入界面控件，启用视频界面控件
        self._set_input_ui_state(False)
        self._set_video_ui_state(True)

        # 切换 UI 框架
        self.input_frame.pack_forget() # 隐藏输入框架
        self.video_frame.pack(fill=tk.BOTH, expand=True) # 显示视频框架

        # 获取视频显示区域的当前尺寸，以便相应调整图片大小
        self.update_idletasks() # 处理待定的几何更新
        video_label_width = self.video_label.winfo_width()
        video_label_height = self.video_label.winfo_height()
        video_label_size = (video_label_width, video_label_height)
        print(f"Video label size: {video_label_size}") # 调试信息

        # 清除停止事件，并启动工作线程
        stop_event.clear()
        global processing_running, current_frame_index, current_frame_data
        processing_running = True # 设置运行标志
        current_frame_index = -1 # 重置帧索引
        current_frame_data = None # 清空共享数据
        # 重置视频显示标签，显示“模型预热中...”文本
        self.video_label.config(image='', text="Model warming up...")

        # 创建工作线程，并将滑动条变量和状态变量作为参数传递
        self.processing_thread = threading.Thread(
            target=process_video_frames,
            args=(yaml_path, video_folder, self.conf_var, self.iou_var, self.status_var, video_label_size)
        )
        # 设置线程为守护线程，允许主线程在工作线程仍在运行时退出
        self.processing_thread.daemon = True
        self.processing_thread.start() # 启动线程

        # 启动 UI 更新循环
        self._schedule_update()

    def _stop_processing(self):
        """向工作线程发送停止信号并等待其完成"""
        global processing_running
        if processing_running: # 如果处理正在运行
            stop_event.set() # 设置事件，向工作线程发送停止信号

            # 短暂等待线程确认并停止
            if self.processing_thread and self.processing_thread.is_alive():
                 print("Waiting for processing thread to finish...")
                 # 设置合理的超时时间，避免无限期阻塞 UI
                 self.processing_thread.join(timeout=5)
                 if self.processing_thread.is_alive():
                      print("Warning: Processing thread did not stop within timeout.")
                      # 您可能需要处理这种情况，例如强制退出？
                      # 目前只打印警告并继续。

            processing_running = False # 设置运行标志为 False
            self.status_var.set("Status: Stopped.") # 更新 UI 状态

            # 停止时清理全局模型和 CUDA 缓存
            global model
            if model is not None:
                del model
                model = None
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()

    def _stop_and_reselect(self):
        """停止处理并切换回输入界面"""
        self._stop_processing() # 停止工作线程

        # 清空显示的图片和状态文本
        self.current_tk_image = None
        self.video_label.config(image='', text="Select video folder and YAML.")
        self.status_var.set("Status: Ready.")

        # 切换 UI 框架
        self.video_frame.pack_forget() # 隐藏视频框架
        self.input_frame.pack(fill=tk.BOTH, expand=True) # 显示输入框架

        # 启用输入界面控件
        self._set_input_ui_state(True)
        self._set_video_ui_state(False) # 禁用视频界面控件


    def _schedule_update(self):
        """调度下一次 UI 更新检查"""
        # 每 30 毫秒检查一次更新 (根据需要调整以平衡响应性和 CPU 使用率)
        self._update_frame_display_scheduled = self.after(30, self._update_frame_display)

    def _update_frame_display(self):
        """检查工作线程是否有新的帧数据，并更新 UI"""
        global current_frame_data, processing_running, current_frame_index, image_files

        # 检查是否有新的数据可用
        if current_frame_data is not None:
            img_tk, status_text = current_frame_data # 获取数据
            self.current_tk_image = img_tk # 保持对 PhotoImage 的引用，防止被垃圾回收！
            # 更新图片和状态文本
            self.video_label.config(image=self.current_tk_image, text="") # 更新图片，清空“预热中”文本
            self.status_var.set(status_text) # 更新状态文本

            # 使用后清空共享数据，表示已消费
            current_frame_data = None

        # 检查工作线程是否仍在运行
        # 也检查处理是否完成 (索引 >= 总帧数)
        # 如果处理仍在运行，或者处理已完成但不是因为停止事件
        if processing_running or (current_frame_index != -1 and current_frame_index < len(image_files) and not stop_event.is_set()):
             # Reschedule the next update
             # 如果处理正在进行，或者处理未完成且未收到停止信号，则继续调度更新
             # 这里的条件可能需要微调，确保在处理完成或停止后不再调度
             # 更简单的检查是：如果 processing_running 为 True，则调度
             if processing_running:
                self._update_frame_display_scheduled = self.after(30, self._update_frame_display)
             # else: processing_running == False, thread finished or stopped, stop scheduling
        elif current_frame_index != -1 and current_frame_index >= len(image_files) - 1:
             # 处理已完成 (工作线程循环结束)
             print("Processing thread finished.")
             self._update_frame_display_scheduled = None # 取消调度更新
             self.status_var.set("Status: Processing complete.") # 更新最终状态
             # 处理完成后清理全局模型和 CUDA 缓存
             global model
             if model is not None:
                 del model
                 model = None
             if torch.cuda.is_available():
                  torch.cuda.empty_cache()
             # (可选) 处理完成后自动切换回输入界面
             # self._stop_and_reselect() # 取消注释以实现自动返回

    def on_closing(self):
        """处理窗口关闭事件"""
        print("Closing application...")
        self._stop_processing() # 向工作线程发送停止信号
        self.destroy() # 销毁 Tkinter 窗口


# ---- 主执行块 ----
if __name__ == "__main__":
    # 创建 App 实例并启动 Tkinter 事件循环
    app = App()
    app.mainloop()
