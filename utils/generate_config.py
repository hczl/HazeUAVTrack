import os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
# 安全读取注释内容
def get_comment(cm: CommentedMap, key: str) -> str:
    try:
        item = cm.ca.items.get(key)
        if item and item[2]:  # item[2] 是“前注释”
            return item[2].value
    except Exception:
        pass
    return None

# 安全设置注释（可强制覆盖）
def set_comment(cm: CommentedMap, key: str, comment: str):
    if key not in cm or not hasattr(cm, 'ca'):
        return
    cm.yaml_set_comment_before_after_key(key, before=comment)
    if key in cm.ca.items:
        if cm.ca.items[key][2] is not None:
            cm.ca.items[key][2].value = comment


# 定义支持的选项范围
SUPPORTED = {
    "haze": ["None", "MiDaS_Deep"],
    "dehaze": ["None", "DIP", "FALCON", "AOD_NET", "BDN"],
    "detector": ["None", "YOLOV3", "IA_YOLOV3", "TDN"],
    "tracker": ["None", "SORT"]
}

# 模板配置（含注释与默认值）
def get_template():
    cfg = CommentedMap()
    cfg.yaml_set_comment_before_after_key('device', before='设置GPU设备，"cuda"表示使用NVIDIA GPU进行加速')
    cfg['device'] = 'cuda'
    cfg.yaml_set_comment_before_after_key('detector_flag', before='是否启用检测器（例如目标检测模型）')
    cfg['detector_flag'] = False
    cfg.yaml_set_comment_before_after_key('tracker_flag', before='是否启用目标跟踪器')
    cfg['tracker_flag'] = False

    method = CommentedMap()
    method.yaml_set_comment_before_after_key('haze', before='生成雾图像的方法（None 或 MiDaS_Deep）')
    method['haze'] = 'MiDaS_Deep'
    method.yaml_set_comment_before_after_key('dehaze', before='去雾方法：None、DIP、FALCON、AOD_NET、BDN')
    method['dehaze'] = 'AOD_NET'
    method.yaml_set_comment_before_after_key('detector', before='检测器类型：None、YOLOV3、IA_YOLOV3、TDN')
    method['detector'] = 'YOLOV3'
    method.yaml_set_comment_before_after_key('tracker', before='跟踪器类型：None、SORT')
    method['tracker'] = 'tracker'
    cfg['method'] = method

    cfg.yaml_set_comment_before_after_key('conf_threshold', before='目标检测的置信度阈值')
    cfg['conf_threshold'] = 0.4
    cfg.yaml_set_comment_before_after_key('iou_threshold', before='NMS中的IOU阈值')
    cfg['iou_threshold'] = 0.5

    dataset = CommentedMap()
    dataset.yaml_set_comment_before_after_key('fog_strength', before='加雾强度（0~1）')
    dataset['fog_strength'] = 0.5
    dataset['train_ratio'] = 0.8
    dataset['val_ratio'] = 0.1
    dataset['batch'] = 64
    dataset['is_mask'] = True
    dataset['path'] = './data/UAV-M'
    dataset['data_path'] = 'UAV-benchmark-M'
    dataset['label_path'] = 'UAV-benchmark-MOTD_v1.0/GT'
    dataset['is_clean'] = True
    dataset['shuffle'] = True
    dataset['nums_worker'] = 8
    cfg['dataset'] = dataset

    cfg.yaml_set_comment_before_after_key('save_output', before='是否保存检测/跟踪结果图像')
    cfg['save_output'] = True
    cfg.yaml_set_comment_before_after_key('seed', before='随机种子（保证结果可复现）')
    cfg['seed'] = 2025

    train = CommentedMap()
    train.yaml_set_comment_before_after_key('checkpoint_interval', before='每N个epoch保存模型')
    train['checkpoint_interval'] = 5
    train['log_interval'] = 10
    train['debug'] = False
    train['epochs'] = 20
    train['resume_training'] = True
    train['lr'] = 0.0001
    train['freeze_dehaze'] = False
    train['pretrain_flag'] = False
    train.yaml_set_comment_before_after_key('dehaze_epoch', before='前N轮仅训练去雾网络')
    train['dehaze_epoch'] = 5
    cfg['train'] = train

    return cfg

# 合并模板，并校验 method 字段合法性
# 合并模板并校验方法合法性，同时强制同步注释
def merge_and_validate(target, template):
    for key, value in template.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_and_validate(target[key], value)

        # 同步注释（即使键已存在也更新）
        if isinstance(template, CommentedMap) and key in template.ca.items:
            comment = get_comment(template, key)
            if comment:
                set_comment(target, key, comment)

    # 校验 method 字段
    if 'method' in target:
        for subkey in ['haze', 'dehaze', 'detector', 'tracker']:
            if subkey in target['method']:
                val = target['method'][subkey]
                if val not in SUPPORTED[subkey]:
                    default_val = template['method'][subkey]
                    print(f"[警告] method.{subkey} 值 '{val}' 非法，重置为默认值 '{default_val}'")
                    target['method'][subkey] = default_val
                # 同步 method 字段下的注释
                if subkey in template['method'].ca.items:
                    comment = get_comment(template, key)
                    if comment:
                        set_comment(target, key, comment)


# 批量更新 YAML 文件并覆盖原路径
def update_yaml_dir(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.load(f) or CommentedMap()
            template = get_template()
            merge_and_validate(data, template)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
            print(f"[完成] 已覆盖更新：{filename}")

# 替换为你配置文件所在目录
update_yaml_dir('../configs')
