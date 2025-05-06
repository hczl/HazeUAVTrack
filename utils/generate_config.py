import os
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# 所有注释集中存储
COMMENTS = {}

def strip_comments(obj, prefix=''):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        for key in keys:
            full_key = f"{prefix}.{key}" if prefix else key
            # 备份注释
            if hasattr(obj, 'ca') and hasattr(obj.ca, 'items'):
                comment_entry = obj.ca.items.get(key)
                if comment_entry and comment_entry[2]:
                    COMMENTS[full_key] = comment_entry[2].value.strip()
            # 递归处理嵌套字典
            if isinstance(obj[key], dict):
                strip_comments(obj[key], prefix=full_key)

        # 清除当前字典的所有注释（安全处理）
        if hasattr(obj, 'ca'):
            if hasattr(obj.ca, 'items'):
                obj.ca.items.clear()
            if hasattr(obj.ca, 'comment'):
                obj.ca.comment = None
            if hasattr(obj.ca, 'end'):
                obj.ca.end = None


# 设置字段但不写入注释
def set_field(cm: CommentedMap, key: str, value, comment: str = None, prefix=''):
    cm[key] = value
    if comment:
        full_key = f"{prefix}.{key}" if prefix else key
        COMMENTS[full_key] = comment

SUPPORTED = {
    "haze": ["None", "MiDaS_Deep"],
    "dehaze": ["None", "DIP", "FALCON", "AOD_NET", "BDN"],
    "detector": ["None", "YOLOV3", "IA_YOLOV3", "TDN"],
    "tracker": ["None", "SORT"],
    "track_method": ["boosttrack", "botsort", "bytetrack", "strongsort", "deepocsort", "ocsort", "imprassoc"]
}

def get_template():
    cfg = CommentedMap()
    set_field(cfg, 'device', 'cuda', '设置GPU设备，"cuda"表示使用NVIDIA GPU进行加速')
    set_field(cfg, 'detector_flag', False, '是否启用检测器（例如目标检测模型）')
    set_field(cfg, 'tracker_flag', False, '是否启用目标跟踪器')

    method = CommentedMap()
    prefix = 'method'
    set_field(method, 'haze', 'MiDaS_Deep', '生成雾图像的方法（None 或 MiDaS_Deep）', prefix)
    set_field(method, 'dehaze', 'AOD_NET', '去雾方法：None、DIP、FALCON、AOD_NET、BDN', prefix)
    set_field(method, 'detector', 'YOLOV3', '检测器类型：None、YOLOV3、IA_YOLOV3、TDN', prefix)
    set_field(method, 'tracker', 'tracker', None, prefix)  # 原注释不保留
    set_field(method, 'track_method', 'bytetrack', '跟踪算法，可选: boosttrack, botsort, bytetrack, strongsort, deepocsort, ocsort, imprassoc', prefix)
    cfg['method'] = method

    set_field(cfg, 'conf_threshold', 0.4, '目标检测的置信度阈值')
    set_field(cfg, 'iou_threshold', 0.5, 'NMS中的IOU阈值')

    dataset = CommentedMap()
    prefix = 'dataset'
    set_field(dataset, 'fog_strength', 0.5, '加雾强度（0~1）', prefix)
    set_field(dataset, 'train_ratio', 0.8, None, prefix)
    set_field(dataset, 'val_ratio', 0.1, None, prefix)
    set_field(dataset, 'batch', 64, None, prefix)
    set_field(dataset, 'is_mask', True, None, prefix)
    set_field(dataset, 'path', './data/UAV-M', None, prefix)
    set_field(dataset, 'data_path', 'UAV-benchmark-M', None, prefix)
    set_field(dataset, 'label_path', 'UAV-benchmark-MOTD_v1.0/GT', None, prefix)
    set_field(dataset, 'is_clean', True, None, prefix)
    set_field(dataset, 'shuffle', True, None, prefix)
    set_field(dataset, 'nums_worker', 8, None, prefix)
    cfg['dataset'] = dataset

    set_field(cfg, 'save_output', True, '是否保存检测/跟踪结果图像')
    set_field(cfg, 'seed', 2025, '随机种子（保证结果可复现）')

    train = CommentedMap()
    prefix = 'train'
    set_field(train, 'checkpoint_interval', 5, '每N个epoch保存模型', prefix)
    set_field(train, 'log_interval', 10, None, prefix)
    set_field(train, 'debug', False, None, prefix)
    set_field(train, 'epochs', 20, None, prefix)
    set_field(train, 'resume_training', True, None, prefix)
    set_field(train, 'lr', 0.0001, None, prefix)
    set_field(train, 'freeze_dehaze', False, None, prefix)
    set_field(train, 'pretrain_flag', False, None, prefix)
    set_field(train, 'dehaze_epoch', 5, '前N轮仅训练去雾网络', prefix)
    cfg['train'] = train

    return cfg

def merge_and_validate(target, template):
    for key, value in template.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_and_validate(target[key], value)

    if 'method' in target:
        for subkey in ['haze', 'dehaze', 'detector', 'tracker', 'track_method']:
            if subkey in target['method']:
                val = target['method'][subkey]
                if val not in SUPPORTED[subkey]:
                    default_val = template['method'][subkey]
                    print(f"[警告] method.{subkey} 值 '{val}' 非法，重置为默认值 '{default_val}'")
                    target['method'][subkey] = default_val

def update_yaml_dir(directory, comment_out_path=None):
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.load(f) or CommentedMap()
            strip_comments(data)
            template = get_template()
            merge_and_validate(data, template)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
            print(f"[完成] 已清除并覆盖注释：{filename}")

    if comment_out_path:
        with open(comment_out_path, 'w', encoding='utf-8') as f:
            json.dump(COMMENTS, f, ensure_ascii=False, indent=2)
        print(f"[完成] 注释已另存至：{comment_out_path}")

# 执行
update_yaml_dir('../configs', comment_out_path='../configs/stripped_comments.json')
