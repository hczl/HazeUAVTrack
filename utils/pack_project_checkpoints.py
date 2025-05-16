import os
import shutil
import json
import argparse
import tempfile
import tarfile

def pack_checkpoints(project_root, output_archive, search_subdirs=None, file_extensions=None):
    project_root_abs = os.path.abspath(project_root)

    if not os.path.isdir(project_root_abs):
        print(f"错误: 项目根目录 '{project_root_abs}' 不存在或不是一个目录。")
        return

    if not search_subdirs:
        print("未指定要扫描的子目录，脚本将退出。")
        print("请使用 --search-subdirs 参数指定，例如 --search-subdirs models/dehaze models/detector")
        return

    print(f"正在扫描项目目录: {project_root_abs}")
    print(f"将在以下子目录中查找模型文件: {search_subdirs}")
    print(f"只打包以下扩展名的文件: {file_extensions}" if file_extensions else "打包所有文件")

    manifest = {}
    files_to_pack = []
    packed_count = 0
    staging_dir = None

    temp_parent_dir = os.path.join(project_root_abs, "__temp")
    staging_dir_prefix = "_project_ckpt_staging_"

    try:
        os.makedirs(temp_parent_dir, exist_ok=True)
        print(f"确保临时目录父目录存在: {temp_parent_dir}")

        staging_dir = tempfile.mkdtemp(prefix=staging_dir_prefix, dir=temp_parent_dir)
        print(f"创建临时暂存目录: {staging_dir}")

        for sub_dir_rel in search_subdirs:
            search_dir_full = os.path.join(project_root_abs, sub_dir_rel)

            if not os.path.isdir(search_dir_full):
                print(f"警告: 扫描目录 '{search_dir_full}' 不存在或不是目录，跳过。")
                continue

            print(f"正在扫描子目录: {sub_dir_rel}")

            for dirpath, dirnames, filenames in os.walk(search_dir_full):
                if os.path.realpath(dirpath).startswith(os.path.realpath(staging_dir)):
                    print(f"跳过临时目录内部扫描: {dirpath}")
                    continue

                for filename in filenames:
                    full_original_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(full_original_path, project_root_abs)

                    if file_extensions:
                        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                        if file_ext not in file_extensions:
                            continue

                    new_name_base = relative_path.replace(os.sep, '__').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

                    new_name = new_name_base
                    counter = 1
                    while new_name in manifest:
                         new_name = f"{new_name_base}_v{counter}"
                         counter += 1

                    target_path_in_staging = os.path.join(staging_dir, new_name)

                    try:
                        shutil.copy2(full_original_path, target_path_in_staging)
                        manifest[new_name] = relative_path
                        files_to_pack.append(relative_path)
                        packed_count += 1
                    except Exception as e:
                        print(f"复制文件失败 '{full_original_path}': {e}")


        if not files_to_pack:
            print("未找到任何符合条件的模型文件。")
            return

        manifest_path_in_staging = os.path.join(staging_dir, "manifest.json")
        with open(manifest_path_in_staging, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted(manifest.items())), f, indent=4, ensure_ascii=False)
        print(f"已生成映射文件: manifest.json")

        output_archive_full_path = os.path.abspath(output_archive)
        print(f"正在创建打包文件: {output_archive_full_path}.tar.gz")
        shutil.make_archive(output_archive_full_path, 'gztar', root_dir=staging_dir, base_dir='.')
        print(f"打包完成: {output_archive_full_path}.tar.gz ({packed_count} 个文件)")

    except Exception as e:
        print(f"打包过程中发生错误: {e}")
    finally:
        if staging_dir and os.path.exists(staging_dir):
            print(f"正在清理临时暂存目录: {staging_dir}")
            try:
                shutil.rmtree(staging_dir)
                print("清理完成。")
            except Exception as e:
                 print(f"清理临时目录失败: {e}")

if __name__ == "__main__":
    default_project_root_arg = None
    default_output_archive = "packed_models"
    default_search_subdirs = ['models/dehaze', 'models/detector']
    default_extensions = ['pth', 'pt', 'ckpt']

    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs='?', default=default_project_root_arg,
                        help=f"项目的根目录路径 (例如 HazeUAVTrack/)。如果未指定，则假定项目根目录是脚本所在目录的父目录。")
    parser.add_argument("output_archive", nargs='?', default=default_output_archive,
                        help=f"输出的压缩文件路径 (不包含扩展名)，例如 'packed_models'。默认为 '{default_output_archive}'。")
    parser.add_argument("--search-subdirs", nargs='+', help=f"相对于 project_root 需要扫描的子目录列表，例如 models/dehaze models/detector。默认为 {default_search_subdirs}",
                        default=default_search_subdirs)
    parser.add_argument("--extensions", nargs='+', help=f"只打包指定扩展名的文件 (不含点)，例如 --extensions pth pt ckpt。默认为 {default_extensions}",
                        default=default_extensions)

    args = parser.parse_args()

    effective_project_root = None
    if args.project_root is None:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        effective_project_root = os.path.dirname(script_dir)
        print(f"未指定项目根目录，根据脚本位置 '{script_path}' 确定为: {effective_project_root}")
    else:
        effective_project_root = os.path.abspath(args.project_root)
        print(f"使用指定的项目根目录: {effective_project_root}")

    pack_checkpoints(effective_project_root, args.output_archive, args.search_subdirs, args.extensions)
