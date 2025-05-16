import os
import shutil
import json
import argparse
import tempfile
import tarfile

def is_within_directory(directory, file_path):
    abs_directory = os.path.abspath(directory)
    abs_file_path = os.path.abspath(file_path)
    try:
        common_path = os.path.commonpath([abs_directory, abs_file_path])
        return common_path == abs_directory
    except ValueError:

        return False

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
            print("正在检查并解压文件...")
            safe_members = []
            for member in tar.getmembers():
                 if member.type in (tarfile.REGTYPE, tarfile.DIRTYPE, tarfile.LNKTYPE, tarfile.SYMTYPE):
                    member_target_path_in_temp = os.path.join(unpack_dir, member.name)
                    if not is_within_directory(unpack_dir, member_target_path_in_temp):
                        print(f"警告: 尝试解压非法路径 '{member.name}'，跳过。")
                        continue
                    safe_members.append(member)
                 else:
                    print(f"跳过特殊文件类型 '{member.name}' (类型: {member.type})。")

            tar.extractall(path=unpack_dir, members=safe_members)
            print("解压完成。")

        manifest_path = os.path.join(unpack_dir, "manifest.json")

        if not os.path.isfile(manifest_path):
            print(f"错误: 在解压目录 '{unpack_dir}' 中未找到 manifest.json 文件。")
            print("请确认打包文件是否正确，且 manifest.json 位于压缩包的根目录。")
            return

        print(f"找到映射文件: manifest.json")
        manifest_data = None
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as e:
             print(f"错误: 解析 manifest.json 文件失败: {e}")
             return
        except Exception as e:
             print(f"错误: 读取 manifest.json 文件失败: {e}")
             return


        if not manifest_data:
            print("映射文件为空，没有模型需要恢复。")
            return

        print(f"正在恢复 {len(manifest_data)} 个模型文件...")
        print(f"目标项目根目录 (绝对路径): {target_project_root_abs}")


        restored_count = 0
        for new_name, original_relative_path in manifest_data.items():
            source_path_in_unpack = os.path.join(unpack_dir, new_name)

            # 核心修改：将manifest中的反斜杠替换为正斜杠，以适应Linux路径规范
            original_relative_path_linux_style = original_relative_path.replace('\\', '/')

            target_full_path = os.path.join(target_project_root_abs, original_relative_path_linux_style)

            print(f"\n处理文件 '{new_name}':")
            print(f"  原始相对路径 (manifest): '{original_relative_path}'")
            print(f"  转换后相对路径: '{original_relative_path_linux_style}'")
            print(f"  计算目标完整路径: '{target_full_path}'")


            if not is_within_directory(target_project_root_abs, target_full_path):
                print(f"警告: 计算目标路径 '{target_full_path}' 不在项目根目录 '{target_project_root_abs}' 内部，跳过恢复。")
                continue

            if not os.path.exists(source_path_in_unpack):
                print(f"警告: 源文件 '{os.path.relpath(source_path_in_unpack, unpack_dir)}' 不存在于临时目录，跳过恢复 '{original_relative_path}'。")
                continue

            target_dir = os.path.dirname(target_full_path)
            if not os.path.exists(target_dir):
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"  创建目标目录: {target_dir}")
                except Exception as e:
                    print(f"  创建目录失败 '{target_dir}': {e}")
                    continue

            try:
                print(f"  正在移动 '{os.path.relpath(source_path_in_unpack, unpack_dir)}' 到 '{os.path.relpath(target_full_path, target_project_root_abs)}'")
                shutil.move(source_path_in_unpack, target_full_path)
                restored_count += 1
            except Exception as e:
                print(f"  恢复文件失败 '{original_relative_path}': {e}")

        print(f"\n模型文件恢复完成。成功恢复 {restored_count} 个文件。")

    except tarfile.TarError as e:
        print(f"错误: 解压tar文件失败: {e}")
    except Exception as e:
        print(f"解包或恢复过程中发生未预期的错误: {e}")
    finally:
        if unpack_dir and os.path.exists(unpack_dir):
            print(f"正在清理临时解压目录: {unpack_dir}")
            try:
                shutil.rmtree(unpack_dir)
                print("清理完成。")
            except Exception as e:
                 print(f"清理临时目录失败: {e}")
        elif unpack_dir:
             print(f"临时目录 '{unpack_dir}' 不存在，无需清理。")


if __name__ == "__main__":
    default_archive_path_arg = None
    default_target_project_root_arg = None

    parser = argparse.ArgumentParser(
        description="解压模型打包文件 (.tar.gz) 并将模型文件恢复到指定的项目根目录。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("archive_path", nargs='?', default=default_archive_path_arg,
                        help="打包文件的路径 (.tar.gz)。\n"
                             f"如果未指定，则默认为当前工作目录下的 'packed_models.tar.gz'。")
    parser.add_argument("target_project_root", nargs='?', default=default_target_project_root_arg,
                        help="远程项目代码的根目录路径 (例如 /path/to/HazeUAVTrack/)，模型将恢复到这里。\n"
                             "请确保这是一个已经传输到远程服务器的项目目录。\n"
                             "如果未指定，则假定项目根目录是脚本所在目录的父目录。")

    args = parser.parse_args()

    effective_archive_path = None
    if args.archive_path is None:
         effective_archive_path = os.path.abspath(os.path.join(os.getcwd(), "packed_models.tar.gz"))
         print(f"未指定打包文件路径，默认为当前工作目录下的: {effective_archive_path}")
    else:
         effective_archive_path = os.path.abspath(args.archive_path)
         print(f"使用指定的打包文件路径: {effective_archive_path}")

    effective_target_project_root = None
    if args.target_project_root is None:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        effective_target_project_root = os.path.dirname(script_dir)
        print(f"未指定目标项目根目录，根据脚本位置 '{script_path}' 确定为脚本所在目录的父目录: {effective_target_project_root}")
    else:
        effective_target_project_root = os.path.abspath(args.target_project_root)
        print(f"使用指定的目标项目根目录: {effective_target_project_root}")

    print("-" * 20)
    unpack_checkpoints(effective_archive_path, effective_target_project_root)
    print("-" * 20)
