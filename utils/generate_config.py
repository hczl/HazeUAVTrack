import os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

CONFIGS_DIR = "../configs"

def generate_config_with_comments(filename="default.yaml"):
    config = CommentedMap()

    config["device"] = "cuda"
    config.yaml_set_comment_before_after_key("device", before="gpu设置")

    config["detector_flag"] = False
    config.yaml_set_comment_before_after_key("detector_flag", before="模型是否启用检测器")

    config["tracker_flag"] = False
    config.yaml_set_comment_before_after_key("tracker_flag", before="模型是否启用追踪器")

    method = CommentedMap()
    method["haze"] = "MiDaS_Deep"
    method.yaml_set_comment_before_after_key("haze", before="生成雾方法，可选: None, MiDaS_Deep")
    method["dehaze"] = "AOD_NET"
    method.yaml_set_comment_before_after_key("dehaze", before="去雾方法，可选: None, DIP, DENET, FALCON, AOD_NET, BDN")
    method["detector"] = "YOLOV3"
    method.yaml_set_comment_before_after_key("detector", before="检测器，可选: YOLOV3,  DITOL")
    method["tracker"] = "tracker"
    method.yaml_set_comment_before_after_key("tracker", before="跟踪算法，可选: SORT")
    config["method"] = method

    config["conf_threshold"] = 0.4
    config["iou_threshold"] = 0.5
    config.yaml_set_comment_before_after_key("conf_threshold", before="检测与跟踪参数")

    dataset = CommentedMap()
    dataset["fog_strength"] = 0.85
    dataset["train_ratio"] = 0.8
    dataset["val_ratio"] = 0.1
    dataset["batch"] = 64
    dataset["is_mask"] = True
    dataset["path"] = "./data/UAV-M"
    dataset["data_path"] = "UAV-benchmark-M"
    dataset["label_path"] = "UAV-benchmark-MOTD_v1.0/GT"
    dataset["is_clean"] = True
    dataset["shuffle"] = True
    config["dataset"] = dataset
    config.yaml_set_comment_before_after_key("dataset", before="数据集路径与设置")

    config["save_output"] = True
    config.yaml_set_comment_before_after_key("save_output", before="是否保存可视化结果")
    config["seed"] = 2025
    config.yaml_set_comment_before_after_key("seed", before="随机种子")

    train = CommentedMap()
    train["checkpoint_interval"] = 5
    train.yaml_set_comment_before_after_key("checkpoint_interval", before="每5个epoch存一次模型")
    train["log_interval"] = 10
    train.yaml_set_comment_before_after_key("log_interval", before="每10个step终端更新一次数据")
    train["debug"] = False
    train["epochs"] = 20
    train["resume_training"] = True
    train["lr"] = 0.0001
    train["freeze_dehaze"] = False
    train["pretrain_flag"] = False
    train["dehaze_epoch"] = 5
    config["train"] = train
    config.yaml_set_comment_before_after_key("train", before="训练参数")

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    with open(os.path.join(CONFIGS_DIR, filename), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    print(f"配置文件生成于: {os.path.join(CONFIGS_DIR, filename)}")

def update_value_in_all_yaml(key_path: str, new_value, new_comment: str = None):
    for fname in os.listdir(CONFIGS_DIR):
        if fname.endswith((".yaml", ".yml")):
            path = os.path.join(CONFIGS_DIR, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f)

            keys = key_path.split('.')
            d = data
            for k in keys[:-1]:
                d = d.get(k, CommentedMap())
            last_key = keys[-1]

            if last_key in d:
                d[last_key] = new_value
                if new_comment:
                    d.yaml_set_comment_before_after_key(last_key, before=new_comment)
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f)
                print(f"{fname} 中已更新 {key_path}")
            else:
                print(f"{fname} 中未找到键：{key_path}")
def update_choice_comment(key_path: str, new_choices: str):
    """
    修改类似 "# 可选: ..." 的注释内容
    key_path: 例如 'method.haze'
    new_choices: 例如 'None, MiDaS_Deep, FogNet'
    """
    for fname in os.listdir(CONFIGS_DIR):
        if not fname.endswith((".yaml", ".yml")):
            continue
        full_path = os.path.join(CONFIGS_DIR, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f)

        keys = key_path.split(".")
        d = data
        for k in keys[:-1]:
            d = d.get(k, CommentedMap())
        last_key = keys[-1]

        if last_key in d:
            # 获取原始注释
            comment_list = d.ca.items.get(last_key)
            if comment_list and comment_list[2]:
                raw_comment = comment_list[2].value.strip()
                if "可选:" in raw_comment:
                    base = raw_comment.split("可选:")[0].strip()
                else:
                    base = raw_comment
                new_comment = f"{base} 可选: {new_choices}"
            else:
                new_comment = f"可选: {new_choices}"
            d.yaml_set_comment_before_after_key(last_key, before=new_comment)
            with open(full_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f)
            print(f"{fname} 中 {key_path} 的可选项已更新为: {new_choices}")
        else:
            print(f"{fname} 中未找到键：{key_path}")
def restructure_existing_yaml(filepath: str, output_name: str = None):
    with open(filepath, 'r', encoding='utf-8') as f:
        old_data = yaml.load(f)

    new_config = CommentedMap()

    def copy_with_comments(dst_map, key, val, comment=None):
        dst_map[key] = val
        if comment:
            dst_map.yaml_set_comment_before_after_key(key, before=comment)

    # ========== 基础设置 ==========
    copy_with_comments(new_config, "device", old_data.get("device", "cuda"), "gpu设置")
    copy_with_comments(new_config, "detector_flag", old_data.get("detector_flag", False), "模型是否启用检测器")
    copy_with_comments(new_config, "tracker_flag", old_data.get("tracker_flag", False), "模型是否启用追踪器")

    # ========== 模型设置 ==========
    method = old_data.get("method", {})
    method_map = CommentedMap()
    copy_with_comments(method_map, "haze", method.get("haze", "MiDaS_Deep"), "生成雾方法，可选: None, MiDaS_Deep")
    copy_with_comments(method_map, "dehaze", method.get("dehaze", "AOD_NET"), "去雾方法，可选: None, DIP, DENET, FALCON, AOD_NET, BDN")
    copy_with_comments(method_map, "detector", method.get("detector", "YOLOV3"), "检测器，可选: YOLOV3,  DITOL")
    copy_with_comments(method_map, "tracker", method.get("tracker", "tracker"), "跟踪算法，可选: SORT")
    new_config["method"] = method_map
    new_config.yaml_set_comment_before_after_key("method", before="模型选择")

    # ========== 检测参数 ==========
    copy_with_comments(new_config, "conf_threshold", old_data.get("conf_threshold", 0.4), "检测与跟踪参数")
    new_config["iou_threshold"] = old_data.get("iou_threshold", 0.5)

    # ========== 数据集设置 ==========
    dataset = old_data.get("dataset", {})
    dataset_map = CommentedMap()
    copy_with_comments(dataset_map, "fog_strength", dataset.get("fog_strength", 0.85), "雾强度")
    dataset_keys = ["train_ratio", "val_ratio", "batch", "is_mask", "path", "data_path", "label_path", "is_clean", "shuffle"]
    for k in dataset_keys:
        dataset_map[k] = dataset.get(k)
    new_config["dataset"] = dataset_map
    new_config.yaml_set_comment_before_after_key("dataset", before="数据集路径与设置")

    # ========== 输出与种子 ==========
    copy_with_comments(new_config, "save_output", old_data.get("save_output", True), "是否保存可视化结果")
    copy_with_comments(new_config, "seed", old_data.get("seed", 2025), "随机种子")

    # ========== 训练参数 ==========
    train = old_data.get("train", {})
    train_map = CommentedMap()
    copy_with_comments(train_map, "checkpoint_interval", train.get("checkpoint_interval", 5), "每5个epoch存一次模型")
    copy_with_comments(train_map, "log_interval", train.get("log_interval", 10), "每10个step终端更新一次数据")
    for key in ["debug", "epochs", "resume_training", "lr", "freeze_dehaze", "pretrain_flag", "dehaze_epoch"]:
        train_map[key] = train.get(key)
    new_config["train"] = train_map
    new_config.yaml_set_comment_before_after_key("train", before="训练参数")

    # ========== 输出 ==========
    out_path = os.path.join(CONFIGS_DIR, output_name or os.path.basename(filepath))
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f)
def restructure_all_yaml_files():
    for fname in os.listdir(CONFIGS_DIR):
        if fname.endswith((".yaml", ".yml")):
            filepath = os.path.join(CONFIGS_DIR, fname)
            print(f"正在重构: {fname}")
            restructure_existing_yaml(filepath)

# 示例使用
if __name__ == "__main__":
    # restructure_all_yaml_files()
    # generate_config_with_comments("my_config.yaml")
    update_value_in_all_yaml("dataset.fog_strength", 0.5, "雾强度")
    # update_choice_comment("method.dehaze", "None, DIP, FALCON, AOD_NET, BDN")
