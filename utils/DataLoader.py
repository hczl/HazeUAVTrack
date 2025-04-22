import os
import yaml
import random
import importlib.util
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class UAVOriginalDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        label_path = self.label_files[idx]
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    annotations.append(line.strip().split(','))
        if self.transform:
            img = self.transform(img)
        return img, annotations


class UAVDataLoaderBuilder:
    def __init__(self, config):
        self.seed = config['seed']
        self.root = config['dataset']['path']
        self.haze_method = config.get('haze_method', 'None')
        self.dehaze_method = config.get('dehaze_method', 'None')

        self.image_root = os.path.join(self.root, config['dataset']['data_path'])
        self.label_root = os.path.join(self.root, config['dataset']['label_path'])
        random.seed(self.seed)

        # 处理 haze/dehaze 方法
        self.image_root = self.apply_processing(self.image_root)

    def apply_processing(self, dataset_path):
        path = dataset_path
        if self.haze_method != "None":
            haze_path = f"./haze/{self.haze_method}.py"
            path = self.call_processing_function(haze_path, "haze", path)


        return path

    def call_processing_function(self, script_path, func_name, input_path):
        assert os.path.exists(script_path), f"Script not found: {script_path}"
        spec = importlib.util.spec_from_file_location("process_module", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = getattr(mod, func_name)
        return func(input_path)

    def parse_labels_to_frames(self, label_file):
        frame_map = {}
        with open(label_file, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                frame = int(items[0])
                frame_map.setdefault(frame, []).append(line.strip())
        return frame_map

    def extract_labels(self, video_folder, label_path, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        frame_map = self.parse_labels_to_frames(label_path)
        image_files = sorted(glob(os.path.join(video_folder, '*.jpg')))
        label_files = []

        for i, img_path in enumerate(image_files, start=1):
            frame_name = os.path.basename(img_path).replace('.jpg', '.txt')
            label_file = os.path.join(save_folder, frame_name)
            with open(label_file, 'w') as f:
                for line in frame_map.get(i, []):
                    f.write(line + '\n')
            label_files.append(label_file)
        return image_files, label_files

    def build(self, train_ratio=0.7, val_ratio=0.2):
        video_folders = sorted(os.listdir(self.image_root))
        random.shuffle(video_folders)
        print(video_folders)
        n = len(video_folders)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set = video_folders[:train_end]
        val_set = video_folders[train_end:val_end]
        test_set = video_folders[val_end:]
        datasets = {}
        for split_name, video_list in [('train', train_set), ('val', val_set), ('test', test_set)]:
            all_imgs, all_labels = [], []
            for vid in video_list:
                vid_folder = os.path.join(self.image_root, vid)
                label_file = os.path.join(self.label_root, f"{vid}_gt_whole.txt")
                label_save_folder = os.path.join(self.root, 'frame_labels', split_name, vid)
                imgs, labels = self.extract_labels(vid_folder, label_file, label_save_folder)
                all_imgs.extend(imgs)
                all_labels.extend(labels)
            datasets[split_name] = UAVOriginalDataset(all_imgs, all_labels)
        return datasets['train'], datasets['val'], datasets['test']
