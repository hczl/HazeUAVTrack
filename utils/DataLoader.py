import os
import sys
import cv2
import pandas as pd
import torch
import yaml
import random
import importlib.util
from glob import glob
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageDraw
import shutil
import time
from pathlib import Path # 更方便地处理路径和创建空文件
import re # 用于从文件名提取帧号
import traceback # 用于打印完整的错误堆栈

from torchvision import transforms

from .common import call_function


class IndexSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
# custom_collate_fn 需要修改以处理潜在的第三个返回值，并期望 ignore 和 label 格式一致
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of images, variable-length label lists,
    and optional variable-length ignore lists.
    Expects labels and ignores (if present) to have the same annotation format length.
    """
    # 检查批次中的第一个样本是否包含忽略数据 (长度是 3)
    has_ignores = len(batch[0]) == 3

    if has_ignores:
        images, labels, ignores = zip(*batch)
    else:
        images, labels = zip(*batch)
        ignores = None  # 或者可以创建一个空的ignores列表

    # 堆叠图像
    images = torch.stack(images, dim=0)

    label_format_size = 9
    labels = [label.detach().clone().float() if isinstance(label, torch.Tensor)
              else torch.empty((0, label_format_size), dtype=torch.float32) for label in labels]

    # 处理忽略列表 (如果存在)
    if has_ignores and ignores is not None:

        ignore_format_size = 9
        # 确保每个元素都是Tensor并进行深拷贝
        ignores = [ignore.detach().clone().float() if isinstance(ignore, torch.Tensor)
                   else torch.empty((0, ignore_format_size), dtype=torch.float32) for ignore in ignores]


        return images, labels, ignores
    else:
        return images, labels


# UAVOriginalDataset 需要修改以接收 is_mask 参数和读取忽略文件 (现在格式与标签文件相同)
class UAVOriginalDataset(Dataset):
    """
    Dataset class for loading UAV image frames (original or pre-masked) and annotations.
    Optionally loads and returns ignore region annotations based on is_mask flag.
    Labels and Ignore annotations are expected to be in the same format.
    """
    def __init__(self, image_files, label_files, ignore_files, transform=None, is_mask=False):
        self.image_files = image_files
        self.label_files = label_files
        self.ignore_files = ignore_files # 存储帧级忽略文件路径
        self.transform = transform
        self.is_mask = is_mask # 控制是否加载和返回忽略信息
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"错误: 图像文件未找到: {img_path}")
            img = Image.new('RGB', (224, 224), color='black')
        except Exception as e:
            print(f"错误: 加载图像文件失败 {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color='black')

        # --- 加载主标签 ---
        label_path = self.label_files[idx]
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    try:
                        fields = list(map(float, line.split(',')))
                        annotations.append(fields)
                    except (ValueError, IndexError):
                        pass

        # --- 加载忽略标签 ---
        ignore_annotations = []
        if self.is_mask:
            ignore_path = self.ignore_files[idx]
            if os.path.exists(ignore_path):
                with open(ignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'): continue
                        try:
                            fields = list(map(float, line.split(',')))
                            ignore_annotations.append(fields)
                        except (ValueError, IndexError):
                            pass

        # === 缩放处理 ===
        original_w, original_h = img.size
        new_w, new_h = 1024, 512
        scale_factor_w = new_w / original_w
        scale_factor_h = new_h / original_h

        img = transforms.functional.resize(img, (new_h, new_w))

        if annotations:
            annotations = torch.tensor(annotations, dtype=torch.float32)
            annotations[:, 2] *= scale_factor_w
            annotations[:, 3] *= scale_factor_h
            annotations[:, 4] *= scale_factor_w
            annotations[:, 5] *= scale_factor_h
        else:
            annotations = torch.empty((0, 9), dtype=torch.float32)  # 假设6列

        if self.is_mask:
            if ignore_annotations:
                ignore_annotations = torch.tensor(ignore_annotations, dtype=torch.float32)
                ignore_annotations[:, 2] *= scale_factor_w
                ignore_annotations[:, 3] *= scale_factor_h
                ignore_annotations[:, 4] *= scale_factor_w
                ignore_annotations[:, 5] *= scale_factor_h
            else:
                ignore_annotations = torch.empty((0, annotations.shape[1]), dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        return (img, annotations, ignore_annotations) if self.is_mask else (img, annotations)


# UAVDataLoaderBuilder 需要修改以加载所有忽略数据并生成帧级忽略文件 (格式与标签相同)
class UAVDataLoaderBuilder:
    """
    构建 UAV 数据集的 DataLoaders，处理数据分割、可选的掩膜预处理（生成帧级忽略文件）、
    可选的雾化/去雾处理以及标签提取。
    现在会为每一帧创建其对应的 ignore 标签文件，格式与主标签文件一致。
    """
    def __init__(self, config):
        """
        使用配置设置初始化构建器。
        """
        self.cfg = config
        self.seed = config['seed']
        self.root = config['dataset']['path']
        self.haze_method = config['method']['haze']
        self.is_mask = config['dataset']['is_mask'] # 保留这个配置，用于控制是否加载和返回忽略信息
        self.is_clean = config['dataset'].get('is_clean', False) # 默认为 False
        self.fog_strength = config['dataset'].get('fog_strength', 0.5) # 默认为 0.5
        # 定义路径
        self.original_image_root = os.path.join(self.root, config['dataset']['data_path'])
        self.label_root = os.path.join(self.root, config['dataset']['label_path'])

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        print(f"初始化 DataLoader Builder...")
        print(f"  原始图像根目录: {self.original_image_root}")
        print(f"  标签根目录: {self.label_root}")
        print(f"  是否加载忽略信息 (is_mask): {self.is_mask}")
        print(f"  雾化方法: {self.haze_method},  雾强度: {self.fog_strength}")

        # --- 加载所有忽略边界框数据到帧级映射 ---
        # 这段数据用于后续生成帧级忽略文件，现在存储原始行字符串
        self.all_frame_ignores_map = self._parse_all_ignore_to_frames()
        print(f"  从 _gt_ignore.txt 文件加载了忽略数据映射。")


        # --- 应用雾化/去雾处理 ---
        # image_root 现在是经过处理后的图像目录
        self.image_root = self.apply_processing(self.original_image_root)
        print(f"  经过处理后的最终图像根目录: {self.image_root}")


    def _parse_all_ignore_to_frames(self):
        """
        加载所有 _gt_ignore.txt 文件，并将忽略边界框数据按视频和帧索引存储到字典中。
        存储原始的行字符串，以便后续直接写入帧级文件。
        实际文件格式为:
        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>,...
        """
        print(f"正在从 {self.label_root} 解析所有 _gt_ignore.txt 数据到帧映射...")
        # 字典结构: {(video_folder, frame_index): [line1, line2, ...]}
        frame_ignore_map = {}

        if not os.path.exists(self.label_root):
            print(f"警告: 标签根目录未找到: {self.label_root}。将不会加载忽略数据。")
            return frame_ignore_map

        ignore_files_list = glob(os.path.join(self.label_root, '*_gt_ignore.txt'))

        if not ignore_files_list:
             print(f"警告: 在 {self.label_root} 中未找到任何 *_gt_ignore.txt 文件。将不会加载忽略数据。")
             return frame_ignore_map

        for ignore_file_path in ignore_files_list:
            file_name = os.path.basename(ignore_file_path)
            vid_folder_name = file_name.replace('_gt_ignore.txt', '')
            # print(f"  正在解析忽略文件: {ignore_file_path} (对应视频: {vid_folder_name})")

            try:
                with open(ignore_file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):  # 跳过空行或注释行
                            continue
                        parts = line.split(',')
                        # 至少需要 frame_index (parts[0])
                        if len(parts) >= 1:
                            try:
                                # frame_index 是第一个元素 (parts[0])
                                frame_index = int(float(parts[0])) # 使用float转换以处理可能的浮点数帧索引
                                # 将原始行添加到映射中
                                key = (vid_folder_name, frame_index)
                                if key not in frame_ignore_map:
                                    frame_ignore_map[key] = []
                                frame_ignore_map[key].append(line) # 存储原始行

                            except ValueError:
                                # print(f"警告: 跳过忽略文件 {ignore_file_path} 中格式错误的行 (非数字帧索引) {line_num}: {line}")
                                pass # 跳过包含非数字帧索引的行
                            except IndexError:
                                # This should theoretically not happen if len(parts) >= 1, but kept for robustness
                                # print(f"警告: 跳过忽略文件 {ignore_file_path} 中列数不足的行 {line_num}: {line}")
                                pass # 跳过列数不足的行

                        else:
                            # print(f"警告: 跳过忽略文件 {ignore_file_path} 中列数不足的行 {line_num}: {line}")
                            pass # 跳过列数不足的行 (少于1列，虽然不太可能)

            except Exception as e:
                print(f"错误: 读取或解析忽略文件 {ignore_file_path} 失败: {e}")
                traceback.print_exc() # 打印更详细的错误信息
                # 继续处理其他文件

        print(f"成功从所有 _gt_ignore.txt 文件解析忽略数据到 {len(frame_ignore_map)} 个帧。")
        return frame_ignore_map


    def apply_processing(self, dataset_path):
        """Applies haze or dehaze processing by calling external functions."""
        path = dataset_path
        if self.haze_method != "NONE":
            print(f"尝试应用雾化方法 '{self.haze_method}'...")
            # 传递雾强度参数给处理函数
            path = call_function(self.haze_method, 'models.haze', path, self.fog_strength, self.cfg['dataset']['nums_worker'])
        return path

    # parse_labels_to_frames 保持不变，用于主标签
    def parse_labels_to_frames(self, label_file):
        """Parses a ground truth file into a dictionary mapping frame numbers to annotations (raw lines)."""
        frame_map = {}
        if not os.path.exists(label_file):
            # print(f"警告: 标签文件未找到: {label_file}")
            return frame_map
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                try:
                    items = line.split(',')
                    # 假设帧号是第一个字段
                    if not items or not items[0].isdigit():
                         # print(f"警告: 跳过标签文件中无效帧格式的行 {label_file}: {line}")
                         continue
                    frame = int(items[0])
                    frame_map.setdefault(frame, []).append(line) # 存储原始行
                except (ValueError, IndexError):
                    # print(f"警告: 跳过标签文件中无效行格式的行 {label_file}: {line}")
                    continue
        return frame_map

    # 修改 extract_labels 以使用帧级忽略映射生成帧级忽略文件 (格式与标签相同)
    def extract_labels(self, video_folder, vid_name, frame_ignore_map, label_path, label_save_folder, ignore_save_folder):
        """
        Extracts frame-specific labels and generates frame-specific ignore files
        using the pre-parsed frame_ignore_map.
        """
        os.makedirs(label_save_folder, exist_ok=True)
        os.makedirs(ignore_save_folder, exist_ok=True) # 创建忽略文件保存目录

        # 解析主标签文件
        frame_label_map = self.parse_labels_to_frames(label_path)

        # 获取所有图像文件
        image_files = sorted(glob(os.path.join(video_folder, '*.jpg')))
        if not image_files:
             # print(f"警告: 在 {video_folder} 中未找到 .jpg 图像")
             return [], [], [] # 返回空列表

        label_files = []
        ignore_files = [] # 这个列表现在将存储帧级忽略文件的路径

        # Helper function to get frame number from filename (robust to different formats)
        def get_frame_num_from_filename(filename):
            # 尝试从文件名中提取数字，通常是最后一个连续的数字串
            # 假设文件名格式是可排序且数字部分代表帧号 (如 000001.jpg, frame_0001.jpg)
            match = re.search(r'\d+', os.path.basename(filename))
            if match:
                 # 找到所有数字串，取最后一个通常是帧号 (如 video_001/frame_00001.jpg -> 00001)
                 all_matches = re.findall(r'\d+', os.path.basename(filename))
                 if all_matches:
                     return int(all_matches[-1])
                 # 如果没有数字串，则返回-1
                 return -1
            else:
                 print(f"警告: 无法从文件名 {os.path.basename(filename)} 提取帧号。跳过此文件。")
                 return -1 # 表示无法确定帧号

        for img_path in image_files:
            frame_base_name = os.path.basename(img_path)
            current_frame_num = get_frame_num_from_filename(frame_base_name)

            if current_frame_num == -1:
                 # 如果无法获取帧号，跳过此图像及其标签/忽略文件生成
                 continue

            # --- 处理主标签文件 ---
            # 使用 Path 对象更方便地处理扩展名
            img_stem = Path(frame_base_name).stem # 获取文件名（不含扩展名）
            label_file_name = img_stem + '.txt'
            label_file_path = os.path.join(label_save_folder, label_file_name)

            with open(label_file_path, 'w') as f:
                # 从解析好的 frame_label_map 中获取当前帧的标签行并写入
                # frame_label_map 键是整数帧号
                for line in frame_label_map.get(current_frame_num, []):
                    f.write(line + '\n')
            label_files.append(label_file_path)

            # --- 生成帧级忽略文件 ---
            ignore_file_name = img_stem + '.txt' # 忽略文件也使用 .txt 扩展名
            ignore_file_path = os.path.join(ignore_save_folder, ignore_file_name)

            # 从预加载的 frame_ignore_map 中获取当前视频和当前帧的忽略行
            # frame_ignore_map 键是 (video_folder, frame_index)
            key = (vid_name, current_frame_num)
            frame_ignores_lines = frame_ignore_map.get(key, []) # 获取原始行列表

            with open(ignore_file_path, 'w') as f:
                if frame_ignores_lines:
                    # 如果找到了忽略行，将原始行直接写入文件 (保持与主标签相同的格式)
                    for line in frame_ignores_lines:
                        f.write(line + '\n')
                # 如果没有忽略行，文件将被创建但为空，这是正确的行为

            ignore_files.append(ignore_file_path) # 添加帧级忽略文件的路径

        # 返回图像文件列表、帧级标签文件列表、帧级忽略文件列表
        return image_files, label_files, ignore_files

    def build(self, train_ratio=0.7, val_ratio=0.2, transform=None):
        """
        Builds train, validation, and test datasets (and optionally a clean dataset).
        Uses the image_root determined in __init__ (original or processed).
        Generates frame-specific label and ignore files before building Datasets.
        Passes the is_mask flag to the Dataset.
        """
        if not os.path.exists(self.image_root):
             raise FileNotFoundError(f"处理后最终的图像根目录未找到: {self.image_root}")

        # 获取经过处理后的图像根目录下的所有视频文件夹
        # 注意：这里假设视频文件夹的名称与原始标签目录下的视频文件夹名称一致
        video_folders_in_processed_root = sorted([d for d in os.listdir(self.image_root) if os.path.isdir(os.path.join(self.image_root, d))])

        if not video_folders_in_processed_root:
             print(f"警告: 在最终图像根目录 {self.image_root} 中未找到视频文件夹。数据集将为空。")
             return None, None, None, None # 返回空的 datasets

        # 使用这些文件夹名称来分割数据集
        random.shuffle(video_folders_in_processed_root)
        n = len(video_folders_in_processed_root)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set_vids = video_folders_in_processed_root[:train_end]
        val_set_vids = video_folders_in_processed_root[train_end:val_end]
        test_set_vids = video_folders_in_processed_root[val_end:]

        datasets = {}
        # 分割并处理每个集合
        for split_name, video_list in [('train', train_set_vids), ('val', val_set_vids), ('test', test_set_vids)]:
            all_imgs, all_labels, all_ignores = [], [], []
            print(f"处理 {split_name} 集，使用来自 {self.image_root} 的图像...")
            if not video_list: # 如果某个 split 的视频列表为空
                print(f"  {split_name} 集中没有分配到视频。")
                datasets[split_name] = None
                continue

            for vid_name in video_list:
                # 图像文件夹路径 (使用处理后的图像根目录)
                vid_folder_path = os.path.join(self.image_root, vid_name)
                # 原始完整标签文件路径
                label_file = os.path.join(self.label_root, f"{vid_name}_gt_whole.txt")

                # 定义帧级标签和忽略文件的保存目录
                # 保存到 self.root 下的 frame_labels 和 frame_ignores 目录
                label_save_folder = os.path.join(self.root, 'frame_labels', split_name, vid_name)
                ignore_save_folder = os.path.join(self.root, 'frame_ignores', split_name, vid_name) # 新增忽略文件保存目录

                # 调用 extract_labels 来生成帧级文件并获取路径列表
                # 将加载的所有忽略数据映射 self.all_frame_ignores_map 传递进去
                imgs, labels, ignores = self.extract_labels(
                    vid_folder_path,
                    vid_name, # 传递视频名称以便在 all_frame_ignores_map 中查询
                    self.all_frame_ignores_map, # 传递加载的忽略数据映射
                    label_file,
                    label_save_folder,
                    ignore_save_folder # 传递忽略文件保存目录
                )

                all_imgs.extend(imgs)
                all_labels.extend(labels)
                all_ignores.extend(ignores)

            if not all_imgs:
                 print(f"警告: 在 {self.image_root} 中未找到 {split_name} 分割的图像，或处理过程中未生成有效数据。")
                 datasets[split_name] = None
            else:
                 # 创建数据集，传递帧级忽略文件列表和 is_mask 标志
                 datasets[split_name] = UAVOriginalDataset(all_imgs, all_labels, all_ignores, transform, is_mask=self.is_mask)
                 print(f"  创建了 {split_name} 数据集，包含 {len(all_imgs)} 个样本。")

        # --- 处理 clean 数据集 (train_clean 和 val_clean) ---
        clean_dataset = None
        val_clean_dataset = None
        if self.is_clean:
            all_imgs_clean, all_labels_clean, all_ignores_clean = [], [], []
            print(f"处理 clean 集，使用来自 {self.original_image_root} 的图像...")

            # 通常 clean 数据集使用训练集的数据
            clean_set_vids = train_set_vids
            if not clean_set_vids:
                print(f"  clean 集中没有分配到视频 (基于 train 集)。")
            else:
                for vid_name in clean_set_vids:
                    vid_folder_path = os.path.join(self.original_image_root, vid_name)
                    label_file = os.path.join(self.label_root, f"{vid_name}_gt_whole.txt")

                    # 定义 clean 集的帧级标签和忽略文件的保存目录
                    label_save_folder = os.path.join(self.root, 'frame_labels', 'clean', vid_name)
                    ignore_save_folder = os.path.join(self.root, 'frame_ignores', 'clean', vid_name)  # clean 集的忽略文件保存目录

                    imgs, labels, ignores = self.extract_labels(
                        vid_folder_path,
                        vid_name,
                        self.all_frame_ignores_map,  # clean 集也使用相同的忽略数据映射
                        label_file,
                        label_save_folder,
                        ignore_save_folder
                    )
                    all_imgs_clean.extend(imgs)
                    all_labels_clean.extend(labels)
                    all_ignores_clean.extend(ignores)
            clean_dataset = UAVOriginalDataset(all_imgs_clean, all_labels_clean, all_ignores_clean, transform,
                                               is_mask=self.is_mask)
            print(f"  创建了 clean 数据集，包含 {len(all_imgs_clean)} 个样本。")

            # val_clean 构建
            all_imgs_val_clean, all_labels_val_clean, all_ignores_val_clean = [], [], []
            print(f"处理 val_clean 集，使用来自 {self.original_image_root} 的图像...")

            val_set_vids_clean = val_set_vids
            if not val_set_vids_clean:
                print(f"  val_clean 集中没有分配到视频。")
            else:
                for vid_name in val_set_vids_clean:
                    vid_folder_path = os.path.join(self.original_image_root, vid_name)
                    label_file = os.path.join(self.label_root, f"{vid_name}_gt_whole.txt")

                    label_save_folder = os.path.join(self.root, 'frame_labels', 'val_clean', vid_name)
                    ignore_save_folder = os.path.join(self.root, 'frame_ignores', 'val_clean', vid_name)

                    imgs, labels, ignores = self.extract_labels(
                        vid_folder_path,
                        vid_name,
                        self.all_frame_ignores_map,
                        label_file,
                        label_save_folder,
                        ignore_save_folder
                    )
                    all_imgs_val_clean.extend(imgs)
                    all_labels_val_clean.extend(labels)
                    all_ignores_val_clean.extend(ignores)

                if not all_imgs_val_clean:
                    print(f"警告: 未找到 val_clean 分割的图像，或处理过程中未生成有效数据。")
                else:
                    val_clean_dataset = UAVOriginalDataset(all_imgs_val_clean, all_labels_val_clean,
                                                           all_ignores_val_clean, transform, is_mask=self.is_mask)
                    print(f"  创建了 val_clean 数据集，包含 {len(all_imgs_val_clean)} 个样本。")

        return datasets.get('train'), datasets.get('val'), datasets.get('test'), clean_dataset, val_clean_dataset


