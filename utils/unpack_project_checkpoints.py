import os
import shutil
import json
import argparse
import tempfile
import tarfile

def unpack_checkpoints(archive_path, target_project_root):
    archive_path_abs = os.path.abspath(archive_path)
    target_project_root_abs = os.path.abspath(target_project_root)

    unpack_dir_prefix = "_temp_project_ckpt_unpack_"
    unpack_dir = None

    if not os.path.isfile(archive_path_abs):
        print(f"错误: 打包文件 '{archive_path_abs}' 不存在。")
        return

    if not os.path.isdir(target_project_root_abs):
        print(f"错误: 目标项目根目录 '{target_project_root_abs}' 不存在或不是一个目录。")
        print("请先将项目代码传输到远程服务器。")
        return

    print(f"正在解压打包文件: {archive_path_abs}")
    print(f"目标项目根目录: {target_project_root_abs}")

    try:
        unpack_dir = tempfile.mkdtemp(prefix=unpack_dir_prefix)
        print(f"创建临时解压目录: {unpack_dir}")

        with tarfile.open(archive_path_abs, 'r:gz') as tar:
            def is_within_directory(directory, file_path):
                abs_directory = os.path.abspath(directory)
                abs_file_path = os.path.abspath(file_path)
                return os.path.commonpath([abs_directory, abs_file_path]) == abs_directory

            print("正在解压...")
            for member in tar.getmembers():
                if member.isfile():
                    member_target_path = os.path.join(unpack_dir, member.name)
                    if not is_within_directory(unpack_dir, member_target_path):
                        print(f"警告: 尝试解压到非法路径，跳过: {member.name}")
                        continue
                    try:
                        os.makedirs(os.path.dirname(member_target_path), exist_ok=True)
                        with open(member_target_path, "wb") as f:
                             shutil.copyfileobj(tar.extractfile(member), f)
                    except Exception as e:
                        print(f"解压文件失败 '{member.name}': {e}")
            print("解压完成。")

        manifest_path = os.path.join(unpack_dir, "manifest.json")

        if not os.path.isfile(manifest_path):
            print(f"错误: 在解压目录 '{unpack_dir}' 中未找到 manifest.json 文件。")
            print("请确认打包文件是否正确，且 manifest.json 位于压缩包的根目录。")
            return

        print(f"找到映射文件: manifest.json")
        manifest_data = None
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)

        if not manifest_data:
            print("映射文件为空，没有模型需要恢复。")
            return

        print(f"正在恢复 {len(manifest_data)} 个模型文件...")

        restored_count = 0
        for new_name, original_relative_path in manifest_data.items():
            source_path_in_unpack = os.path.join(unpack_dir, new_name)
            target_full_path = os.path.join(target_project_root_abs, original_relative_path)

            if not os.path.exists(source_path_in_unpack):
                print(f"警告: 源文件 '{os.path.relpath(source_path_in_unpack, unpack_dir)}' 不存在，跳过恢复 '{original_relative_path}'。")
                continue

            target_dir = os.path.dirname(target_full_path)
            if not os.path.exists(target_dir):
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception as e:
                    print(f"创建目录失败 '{target_dir}': {e}")
                    continue

            try:
                shutil.move(source_path_in_unpack, target_full_path)
                restored_count += 1
            except Exception as e:
                print(f"恢复文件失败 '{original_relative_path}': {e}")

        print(f"模型文件恢复完成。成功恢复 {restored_count} 个文件。")

    except Exception as e:
        print(f"解包或恢复过程中发生错误: {e}")
    finally:
        if unpack_dir and os.path.exists(unpack_dir):
            print(f"正在清理临时解压目录: {unpack_dir}")
            try:
                shutil.rmtree(unpack_dir)
                print("清理完成。")
            except Exception as e:
                print(f"清理临时目录失败: {e}")

if __name__ == "__main__":
    default_archive_path_arg = None
    default_target_project_root_arg = None

    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", nargs='?', default=default_archive_path_arg,
                        help=f"打包文件的路径 (.tar.gz)。如果未指定，则默认为当前目录下的 '{'packed_models.tar.gz'}'。")
    parser.add_argument("target_project_root", nargs='?', default=default_target_project_root_arg,
                        help=f"远程项目代码的根目录路径 (例如 HazeUAVTrack/)，模型将恢复到这里。如果未指定，则假定项目根目录是脚本所在目录的父目录。")

    args = parser.parse_args()

    effective_archive_path = None
    if args.archive_path is None:
         effective_archive_path = os.path.abspath("./packed_models.tar.gz")
         print(f"未指定打包文件路径，默认为: {effective_archive_path}")
    else:
         effective_archive_path = os.path.abspath(args.archive_path)
         print(f"使用指定的打包文件路径: {effective_archive_path}")


    effective_target_project_root = None
    if args.target_project_root is None:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        effective_target_project_root = os.path.dirname(script_dir)
        print(f"未指定目标项目根目录，根据脚本位置 '{script_path}' 确定为: {effective_target_project_root}")
    else:
        effective_target_project_root = os.path.abspath(args.target_project_root)
        print(f"使用指定的目标项目根目录: {effective_target_project_root}")

    unpack_checkpoints(effective_archive_path, effective_target_project_root)
