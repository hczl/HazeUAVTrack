import os
import sys
import torch
import yaml
import random
import importlib.util
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import shutil
import time # 用于计时
from pathlib import Path # 更方便地处理路径和创建空文件

# custom_collate_fn 和 UAVOriginalDataset 保持不变 (从之前的回答复制)
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of images and variable-length label lists.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = [torch.tensor(label, dtype=torch.float32) if label else torch.empty((0, 9), dtype=torch.float32) for label in labels]
    return images, labels

class UAVOriginalDataset(Dataset):
    """
    Dataset class for loading UAV image frames (original or pre-masked) and annotations.
    """
    def __init__(self, image_files, label_files, ignore_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.ignore_files = ignore_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"错误: 图像文件未找到: {img_path}")
            img = Image.new('RGB', (224, 224), color='black') # 返回一个黑色图像作为替代

        label_path = self.label_files[idx]
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        fields = list(map(float, line.strip().split(',')))
                        if len(fields) < 9:
                            # print(f"警告: 跳过标签文件中格式不正确的行 {label_path}: {line.strip()}")
                            continue
                        annotations.append(fields)
                    except ValueError:
                        # print(f"警告: 跳过标签文件中的无效数字数据 {label_path}: {line.strip()}")
                        continue

        if self.transform:
            img = self.transform(img)

        return img, annotations


class UAVDataLoaderBuilder:
    """
    构建 UAV 数据集的 DataLoaders，处理数据分割、可选的掩膜预处理、
    可选的雾化/去雾处理以及标签提取。
    如果启用掩膜，它会预先创建掩膜图像并保存到单独目录。
    如果掩膜目录和完成标记已存在，则跳过掩膜生成。
    """
    def __init__(self, config):
        """
        使用配置设置初始化构建器。
        """
        self.seed = config['seed']
        self.root = config['dataset']['path']
        self.haze_method = config.get('haze_method', "None")
        self.dehaze_method = config.get('dehaze_method', "None")
        self.is_mask = config['dataset']['is_mask']
        self.is_clean = config['dataset']['is_clean']

        # 定义路径
        self.original_image_root = os.path.join(self.root, config['dataset']['data_path'])
        self.label_root = os.path.join(self.root, config['dataset']['label_path'])
        self.masked_image_root = self.original_image_root + "_masked" # 掩膜图片的目录
        # self.masking_complete_marker = os.path.join(self.masked_image_root, ".masking_complete") # 掩膜完成标记文件

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.image_root = self.apply_processing(self.original_image_root)
        print(f"应用雾化/去雾处理后的最终图像根目录: {self.image_root}")
        # --- 掩膜预处理 (检查与执行) ---
        if self.is_mask:
            print(f"配置要求进行掩膜处理。目标目录: '{self.masked_image_root}'")
            # 检查是否已完成
            if os.path.exists(self.masked_image_root):
                print(f"  发现标记文件夹 '{self.masked_image_root}'。跳过掩膜生成环节。")
                self.image_root = self.masked_image_root # 直接使用已存在的掩膜目录
            else:
                print(f"  未发现标记文件或掩膜目录不完整。开始执行掩膜预处理...")
                start_time = time.time()
                try:
                    self._preprocess_apply_masks()
                    # 只有在 _preprocess_apply_masks 成功完成后才设置标记
                    # (该方法内部会创建标记文件)
                    end_time = time.time()
                    print(f"  掩膜预处理完成。耗时: {end_time - start_time:.2f} 秒。")
                    self.image_root = self.masked_image_root # 使用新生成的掩膜目录
                except Exception as e:
                    print(f"错误: 掩膜预处理失败: {e}")
                    print("将尝试使用原始图像，但这可能不是预期行为。")
                    # 出错时回退到原始图像路径
                    self.image_root = self.original_image_root
                    # 可选：如果出错，删除可能不完整的掩膜目录
                    # if os.path.exists(self.masked_image_root):
                    #     print(f"  正在删除可能不完整的掩膜目录: {self.masked_image_root}")
                    #     shutil.rmtree(self.masked_image_root)
        else:
            # 不进行掩膜处理，使用原始图像
            print("配置未要求进行掩膜处理。使用原始图像。")
            self.image_root = self.original_image_root

        print(f"最终用于数据集构建的图像根目录: {self.image_root}")


    def _apply_mask_to_image(self, image_path, ignore_file_path):
        """加载图像，根据忽略文件应用掩膜，返回掩膜后的 PIL 图像。"""
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            if not os.path.exists(ignore_file_path):
                return img  # 没有忽略文件则返回原图

            with open(ignore_file_path, 'r') as f:
                # print(ignore_file_path)
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) == 9:
                            try:
                                bbox_left = int(float(parts[2]))
                                bbox_top = int(float(parts[3]))
                                bbox_width = int(float(parts[4]))
                                bbox_height = int(float(parts[5]))

                                x1 = max(0, min(img_width - 1, bbox_left))
                                y1 = max(0, min(img_height - 1, bbox_top))
                                x2 = max(0, min(img_width - 1, bbox_left + bbox_width))
                                y2 = max(0, min(img_height - 1, bbox_top + bbox_height))

                                draw.rectangle([(x1, y1), (x2, y2)], fill=(128,128,128), outline='black')
                            except ValueError:
                                print(f"警告: 跳过忽略文件中边界框数据的非数字值 {ignore_file_path}: {line.strip()}")
                                continue
                        else:
                            print(f"警告: 忽略文件中行格式不正确 {ignore_file_path}: {line.strip()}")
                            continue
                    except ValueError:
                        print(f"警告: 跳过忽略文件中的非数字数据或格式错误 {ignore_file_path}: {line.strip()}")
                        continue
            return img
        except FileNotFoundError:
            print(f"错误: 掩膜处理时图像文件未找到: {image_path}")
            return None
        except Exception as e:
            print(f"错误: 应用掩膜到 {image_path} (使用 {ignore_file_path}) 时出错: {e}")
            return None


    def _preprocess_apply_masks(self):
        """
        迭代原始图像，应用掩膜，保存到 masked_image_root。
        如果成功完成所有处理，则创建 .masking_complete 标记文件。
        """
        if not os.path.exists(self.original_image_root):
             raise FileNotFoundError(f"原始图像根目录未找到: {self.original_image_root}")

        # 确保目标掩膜目录存在 (即使是空的)
        os.makedirs(self.masked_image_root, exist_ok=True)

        video_folders = sorted([d for d in os.listdir(self.original_image_root) if os.path.isdir(os.path.join(self.original_image_root, d))])
        if not video_folders:
            print("警告: 原始图像目录中没有找到视频子文件夹。")
            # 即使没有视频，也应该创建标记文件表示“处理”完成（虽然是空处理）
            try:
                Path(self.masking_complete_marker).touch()
                print(f"  已创建标记文件 (空目录): {self.masking_complete_marker}")
            except OSError as e:
                print(f"错误: 创建标记文件失败: {e}")
            return # 没有可处理的内容

        total_files_processed = 0
        total_files_skipped = 0
        errors_occurred = False

        for vid_folder_name in video_folders:
            original_vid_path = os.path.join(self.original_image_root, vid_folder_name)
            masked_vid_path = os.path.join(self.masked_image_root, vid_folder_name)
            os.makedirs(masked_vid_path, exist_ok=True) # 创建目标视频目录

            ignore_file = os.path.join(self.label_root, f"{vid_folder_name}_gt_ignore.txt")
            ignore_file_exists = os.path.exists(ignore_file)
            if not ignore_file_exists:
                print(f"  警告: 视频 {vid_folder_name} 的忽略文件未找到: {ignore_file}。此视频中的图像将不会被掩膜，将复制原图。")

            # print(f"  处理视频: {vid_folder_name}...")
            image_files = sorted(glob(os.path.join(original_vid_path, '*.jpg')))
            if not image_files:
                # print(f"  警告: 视频 {vid_folder_name} 中未找到 .jpg 文件。")
                continue # 处理下一个视频

            for img_path in image_files:
                base_name = os.path.basename(img_path)
                masked_img_path = os.path.join(masked_vid_path, base_name)

                # 如果目标文件已存在，可以跳过（可选，增加检查成本 vs 重复写入成本）
                # if os.path.exists(masked_img_path):
                #     total_files_skipped += 1
                #     continue

                if ignore_file_exists:
                    masked_img = self._apply_mask_to_image(img_path, ignore_file)
                    if masked_img:
                        try:
                            masked_img.save(masked_img_path)
                            total_files_processed += 1
                        except Exception as e:
                            print(f"错误: 保存掩膜图像 {masked_img_path} 失败: {e}")
                            errors_occurred = True
                    else:
                        print(f"错误: 无法为 {img_path} 生成掩膜图像。")
                        errors_occurred = True
                        # 决定是否复制原图或跳过
                        try: # 尝试复制原图作为后备
                           shutil.copy2(img_path, masked_img_path)
                           print(f"  -> 已复制原图到 {masked_img_path}")
                        except Exception as copy_e:
                            print(f"错误: 复制原图 {img_path} 到 {masked_img_path} 也失败: {copy_e}")
                else:
                    # 忽略文件不存在，复制原图
                    try:
                        shutil.copy2(img_path, masked_img_path)
                        total_files_processed += 1 # 计为已处理（复制也是一种处理）
                    except Exception as e:
                         print(f"错误: 复制原图 {img_path} 到 {masked_img_path} 失败: {e}")
                         errors_occurred = True

        print(f"  掩膜处理统计: 处理/复制 {total_files_processed} 个文件, 跳过 {total_files_skipped} 个文件。")

        # --- 创建标记文件 ---
        if not errors_occurred:
            try:
                print(f"  所有文件处理完毕，未报告严重错误。正在创建标记文件: {self.masking_complete_marker}")
                Path(self.masking_complete_marker).touch() # 创建空文件
                print(f"  标记文件创建成功。")
            except OSError as e:
                print(f"错误: 创建标记文件 {self.masking_complete_marker} 失败: {e}")
                # 即使标记失败，数据可能已生成，但下次仍会尝试重新生成
        else:
            print(f"警告: 掩膜处理过程中发生错误。标记文件 '{self.masking_complete_marker}' 将不会被创建。下次运行时将尝试重新生成掩膜数据。")

    def apply_processing(self, dataset_path):
        """Applies haze or dehaze processing by calling external functions."""
        path = dataset_path
        if self.haze_method != "None":
            print(f"尝试应用雾化方法 '{self.haze_method}'...")
            path = self.call_processing_function(path, self.haze_method, 'haze')
        return path

    def call_processing_function(self, input_path, method_name, module_prefix):
        """Dynamically imports and calls a processing function."""
        try:
            module_name = f'{module_prefix}.{method_name}'
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                 print(f"错误: 模块 '{module_name}' 未找到。")
                 return input_path
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module # Add to sys.modules before exec
            spec.loader.exec_module(module)

            if not hasattr(module, method_name):
                 print(f"错误: 函数 '{method_name}' 在模块 '{module_name}' 中未找到。")
                 return input_path
            func = getattr(module, method_name)

            print(f"从 {module_prefix} 模块应用 {method_name} 处理到路径: {input_path}")
            output_path = func(input_path)
            if output_path is None:
                 print(f"警告: 处理函数 {module_name}.{method_name} 未返回路径。使用原始路径: {input_path}")
                 return input_path
            print(f"处理函数返回路径: {output_path}")
            return output_path
        except Exception as e:
            print(f"错误: 在 {method_name} 处理 ({module_prefix}) 过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return input_path

    def parse_labels_to_frames(self, label_file):
        """Parses a ground truth file into a dictionary mapping frame numbers to annotations."""
        frame_map = {}
        if not os.path.exists(label_file):
            print(f"警告: 标签文件未找到: {label_file}")
            return frame_map
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    items = line.strip().split(',')
                    if not items or not items[0].isdigit():
                        # print(f"警告: 跳过标签文件中无效帧格式的行 {label_file}: {line.strip()}")
                        continue
                    frame = int(items[0])
                    frame_map.setdefault(frame, []).append(line.strip())
                except (ValueError, IndexError):
                    # print(f"警告: 跳过标签文件中无效行格式的行 {label_file}: {line.strip()}")
                    continue
        return frame_map

    def extract_labels(self, video_folder, label_path, ignore_label_path, save_folder):
        """
        Extracts frame-specific labels and associates ignore files.
        """
        os.makedirs(save_folder, exist_ok=True)
        frame_map = self.parse_labels_to_frames(label_path)
        image_files = sorted(glob(os.path.join(video_folder, '*.jpg')))
        if not image_files:
             # print(f"警告: 在 {video_folder} 中未找到 .jpg 图像")
             pass # 允许空列表返回

        label_files = []
        ignore_files = []

        def get_frame_num_from_filename(filename):
            import re
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else -1

        for img_path in image_files:
            frame_base_name = os.path.basename(img_path)
            label_file_name = frame_base_name.replace('.jpg', '.txt')
            label_file_path = os.path.join(save_folder, label_file_name)

            current_frame_num = get_frame_num_from_filename(frame_base_name)
            if current_frame_num == -1:
                 print(f"警告: 无法从 {frame_base_name} 提取帧号。将为此文件创建空标签。")
                 with open(label_file_path, 'w') as f: pass # 创建空文件
            else:
                with open(label_file_path, 'w') as f:
                    for line in frame_map.get(current_frame_num, []):
                        f.write(line + '\n')

            label_files.append(label_file_path)
            ignore_files.append(ignore_label_path)

        if len(image_files) != len(label_files) or len(image_files) != len(ignore_files):
            print(f"错误: 视频 {video_folder} 的列表长度不匹配。图像: {len(image_files)}, 标签: {len(label_files)}, 忽略: {len(ignore_files)}")
            # 采取纠正措施，例如截断到最短长度
            min_len = min(len(image_files), len(label_files), len(ignore_files))
            image_files = image_files[:min_len]
            label_files = label_files[:min_len]
            ignore_files = ignore_files[:min_len]

        return image_files, label_files, ignore_files

    def build(self, train_ratio=0.7, val_ratio=0.2, transform=None):
        """
        Builds train, validation, and test datasets (and optionally a clean dataset).
        Uses the image_root determined in __init__ (original or masked).
        """
        if not os.path.exists(self.image_root):
             raise FileNotFoundError(f"处理后最终的图像根目录未找到: {self.image_root}")

        video_folders = sorted([d for d in os.listdir(self.image_root) if os.path.isdir(os.path.join(self.image_root, d))])
        if not video_folders:
             # 如果 image_root 存在但是空的，这不一定是错误（例如，原始数据就是空的）
             print(f"警告: 在最终图像根目录 {self.image_root} 中未找到视频文件夹。数据集将为空。")
             return None, None, None, None # 返回空的 datasets

        random.shuffle(video_folders)
        n = len(video_folders)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set_vids = video_folders[:train_end]
        val_set_vids = video_folders[train_end:val_end]
        test_set_vids = video_folders[val_end:]

        datasets = {}
        for split_name, video_list in [('train', train_set_vids), ('val', val_set_vids), ('test', test_set_vids)]:
            all_imgs, all_labels, all_ignores = [], [], []
            print(f"处理 {split_name} 集，使用来自 {self.image_root} 的图像...")
            if not video_list: # 如果某个 split 的视频列表为空
                print(f"  {split_name} 集中没有分配到视频。")
                datasets[split_name] = None
                continue

            for vid_name in video_list:
                vid_folder_path = os.path.join(self.image_root, vid_name)
                label_file = os.path.join(self.label_root, f"{vid_name}_gt_whole.txt")
                ignore_file = os.path.join(self.label_root, f"{vid_name}_gt_ignore.txt")
                label_save_folder = os.path.join(self.root, 'frame_labels', split_name, vid_name)

                imgs, labels, ignores = self.extract_labels(vid_folder_path, label_file, ignore_file, label_save_folder)

                all_imgs.extend(imgs)
                all_labels.extend(labels)
                all_ignores.extend(ignores)

            if not all_imgs:
                 print(f"警告: 在 {self.image_root} 中未找到 {split_name} 分割的图像。")
                 datasets[split_name] = None
            else:
                 datasets[split_name] = UAVOriginalDataset(all_imgs, all_labels, all_ignores, transform)
                 print(f"  创建了 {split_name} 数据集，包含 {len(all_imgs)} 个样本。")

        clean_dataset = None
        if self.is_clean:
            all_imgs_clean, all_labels_clean, all_ignores_clean = [], [], []
            print(f"处理 clean 集，使用来自 {self.image_root} 的图像...")
            clean_set_vids = train_set_vids # 假设 clean 使用与 train 相同的视频源
            if not clean_set_vids:
                print(f"  clean 集中没有分配到视频 (基于 train 集)。")
            else:
                for vid_name in clean_set_vids:
                    vid_folder_path = os.path.join(self.image_root, vid_name)
                    label_file = os.path.join(self.label_root, f"{vid_name}_gt_whole.txt")
                    ignore_file = os.path.join(self.label_root, f"{vid_name}_gt_ignore.txt")
                    label_save_folder = os.path.join(self.root, 'frame_labels', 'clean', vid_name)
                    imgs, labels, ignores = self.extract_labels(vid_folder_path, label_file, ignore_file, label_save_folder)
                    all_imgs_clean.extend(imgs)
                    all_labels_clean.extend(labels)
                    all_ignores_clean.extend(ignores)

                if not all_imgs_clean:
                     print(f"警告: 未找到 clean 分割的图像。")
                else:
                    clean_dataset = UAVOriginalDataset(all_imgs_clean, all_labels_clean, all_ignores_clean, transform)
                    print(f"  创建了 clean 数据集，包含 {len(all_imgs_clean)} 个样本。")

        return datasets.get('train'), datasets.get('val'), datasets.get('test'), clean_dataset


