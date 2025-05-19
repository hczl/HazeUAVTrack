import os
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import traceback
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Process, Manager, set_start_method

from utils.config import load_config
from utils.create import create_model
from utils.metrics import compute_map, compute_f1, compute_mota
from utils.transform import load_annotations, scale_ground_truth_boxes, scale_ignore_regions

set_start_method('spawn', force=True)

# 全局设置
os.environ['TORCH_HOME'] = './.torch'
RESULT_DIR = 'result/detector_combined'
os.makedirs(RESULT_DIR, exist_ok=True)

TRACKER_NAMES = ['boosttrack', 'strongsort', 'deepocsort', 'botsort', 'ocsort']
TRACKER_THRESHOLDS = {name: (0.5, 0.5) for name in TRACKER_NAMES}
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
SOURCE_FOLDERS = ['M1005', 'M0301', 'M1002', 'M1202', 'M0205', 'M1007']
BASE_GT_LABEL = 'data/UAV-M/frame_labels/test'
BASE_IGNORE = 'data/UAV-M/frame_ignores/test'
BASE_FOG = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog'
FOG_STRENGTHS = [0.5, 0.75]
YAML_PATH = 'configs/DE_NET.yaml'
MAX_SIZE = 1024

DETECTOR_DEHAZE = [
    ("AD_YOLOV11", "NONE"), ("DE_NET", "NONE"), ("IA_YOLOV3", "NONE"),
    ("YOLOV3", "NONE"), ("YOLOV11", "NONE"),
    ("YOLOV3", "FFA"), ("YOLOV11", "FFA")
]


def preprocess_image(pil_img):
    w, h = pil_img.size
    img_tensor = transforms.ToTensor()(pil_img)
    r = min(1.0, MAX_SIZE / float(max(w, h)))
    new_h = max(32, int(math.floor(h * r / 32) * 32))
    new_w = max(32, int(math.floor(w * r / 32) * 32))
    resized = F.resize(img_tensor, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
    normed = F.normalize(resized, mean=NORM_MEAN, std=NORM_STD)
    return normed, (w, h), (new_w, new_h)  # 不再 unsqueeze(0)



class InferenceImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor, _, _ = preprocess_image(img)
        return tensor, path


def worker(gpu_id, source_folder, tracker, detector, dehaze, fog_strengths, return_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        cfg = load_config(YAML_PATH)
        cfg['method'] = {
            'detector': detector,
            'dehaze': dehaze,
            'tracker': 'tracker',
            'track_method': tracker,
            'conf_threshold': None
        }
    except Exception:
        traceback.print_exc()
        return

    metrics_per_fog = {}

    for fog in fog_strengths:
        try:
            fog_code = f"{int(fog * 100):03d}"
            img_dir = os.path.join(f"{BASE_FOG}_{fog_code}", source_folder)
            gt_dir = os.path.join(BASE_GT_LABEL, source_folder)
            ig_dir = os.path.join(BASE_IGNORE, source_folder)

            thresh_idx = 0 if fog == 0.5 else 1
            cfg['method']['conf_threshold'] = TRACKER_THRESHOLDS[tracker][thresh_idx]
            cfg['dataset']['fog_strength'] = fog
            model = create_model(cfg)
            model.load_model()
            model.to(device)
            model.eval()

            image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
            image_paths = [os.path.join(img_dir, f) for f in image_files]
            dataset = InferenceImageDataset(image_paths)
            loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

            first_img = Image.open(image_paths[0]).convert("RGB")
            _, orig, resized = preprocess_image(first_img)
            gt_labels, ignore_masks = load_annotations(gt_dir, ig_dir, len(image_paths))
            gt_scaled = scale_ground_truth_boxes(gt_labels, orig, resized)
            ig_scaled = scale_ignore_regions(ignore_masks, orig, resized)

            preds = []
            times = []

            with torch.no_grad():
                for img_tensor, _ in tqdm(loader,
                                          desc=f"[GPU{gpu_id}] {source_folder}-{tracker}-{detector}-{dehaze}-fog{fog}",
                                          position=gpu_id, leave=True):
                    img_tensor = img_tensor.to(device, non_blocking=True)
                    t0 = time.time()
                    out = model.predict(img_tensor)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                    if isinstance(out, torch.Tensor):
                        out = out.cpu().numpy().tolist()
                    elif isinstance(out, np.ndarray):
                        out = out.tolist()
                    boxes = [b for b in out if len(b) > 5 and b[5] >= cfg['method']['conf_threshold']]
                    preds.append(boxes)

            fps = len(times) / sum(times) if sum(times) > 0 else 0
            mAP = compute_map(preds, gt_scaled, ignore_masks=ig_scaled)
            f1 = compute_f1(preds, gt_scaled, ignore_masks=ig_scaled)
            mota, motp, ids = compute_mota(preds, gt_scaled, ignore_masks=ig_scaled)

        except Exception:
            traceback.print_exc()
            fps = mAP = f1 = mota = motp = ids = np.nan

        metrics_per_fog[fog] = (fps, mAP, f1, mota, motp, ids)

    return_list.append(metrics_per_fog)


if __name__ == "__main__":
    manager = Manager()
    for tracker in TRACKER_NAMES:
        summary = {}
        for detector, dehaze in DETECTOR_DEHAZE:
            ret_list = manager.list()
            procs = []
            for gpu_id, folder in enumerate(SOURCE_FOLDERS):
                p = Process(target=worker,
                            args=(gpu_id, folder, tracker,
                                  detector, dehaze, FOG_STRENGTHS,
                                  ret_list))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

            agg = {}
            arr = list(ret_list)
            for fog in FOG_STRENGTHS:
                vals = [v.get(fog, (np.nan,) * 6) for v in arr]
                mat = np.array(vals, dtype=float)
                means = np.nanmean(mat, axis=0)
                agg[fog] = means
            summary[(detector, dehaze)] = agg

        rows = []
        for (det, dh), fogs in summary.items():
            row = [det, dh]
            for fog in FOG_STRENGTHS:
                metrics = fogs[fog]
                row += [f"{metrics[0]:.2f}", f"{metrics[1]:.4f}", f"{metrics[2]:.4f}",
                        f"{metrics[3]:.4f}", f"{metrics[4]:.4f}", f"{metrics[5]:.2f}"]
            rows.append(row)

        cols = ["Detector", "Dehaze"]
        for fog in FOG_STRENGTHS:
            cols += [f"FPS({fog})", f"mAP({fog})", f"F1({fog})",
                     f"MOTA({fog})", f"MOTP({fog})", f"IDS({fog})"]

        df = pd.DataFrame(rows, columns=cols)
        out_path = os.path.join(RESULT_DIR, f"{tracker}_summary.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"[{tracker}] 结果已保存到 {out_path}")

    print("全部追踪器推理完成。")
