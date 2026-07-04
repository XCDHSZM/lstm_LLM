"""
下载 PTB (Penn Treebank) 原始数据
来源: Mikolov's simple-examples
URL: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

如果下载失败，也可以手动下载 simple-examples.tgz 并解压到 data/raw/ 目录。
解压后应包含:
  - data/ptb.train.txt
  - data/ptb.valid.txt
  - data/ptb.test.txt
"""

import os
import sys
import tarfile
import urllib.request

# 添加 src 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import RAW_DATA_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE

# PTB 数据下载 URL
PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"


def download_file(url: str, dest_path: str):
    """下载文件，带进度显示。"""
    print(f"正在下载: {url}")
    print(f"保存到: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size)) if total_size > 0 else 0
        sys.stdout.write(f"\r  下载进度: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print("\n  下载完成!")
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False


def extract_tgz(tgz_path: str, dest_dir: str):
    """解压 .tgz 文件。"""
    print(f"正在解压: {tgz_path}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print(f"  解压完成!")


def check_raw_data():
    """检查原始数据是否已就绪。"""
    expected_files = [TRAIN_FILE, VALID_FILE, TEST_FILE]
    all_exist = True
    for f in expected_files:
        path = os.path.join(RAW_DATA_DIR, f)
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path)
            print(f"  ✓ {f} ({size:,} bytes)")
        else:
            print(f"  ✗ {f} (缺失)")
            all_exist = False
    return all_exist


def main():
    print("=" * 60)
    print("PTB 数据下载工具")
    print("=" * 60)
    print(f"目标目录: {RAW_DATA_DIR}")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # 检查数据是否已存在
    print("\n检查现有数据...")
    if check_raw_data():
        print("\n所有数据文件已就绪，无需下载。")
        return

    # 下载
    print("\n需要下载 PTB 数据集。")
    tgz_path = os.path.join(RAW_DATA_DIR, "simple-examples.tgz")

    if not os.path.exists(tgz_path):
        success = download_file(PTB_URL, tgz_path)
        if not success:
            print("\n" + "=" * 60)
            print("自动下载失败。请手动下载:")
            print(f"  URL: {PTB_URL}")
            print(f"  解压到: {RAW_DATA_DIR}")
            print("  确保以下文件存在:")
            print(f"    - {os.path.join(RAW_DATA_DIR, TRAIN_FILE)}")
            print(f"    - {os.path.join(RAW_DATA_DIR, VALID_FILE)}")
            print(f"    - {os.path.join(RAW_DATA_DIR, TEST_FILE)}")
            print("=" * 60)
            return
    else:
        print(f"找到已下载的压缩包: {tgz_path}")

    # 解压
    extract_tgz(tgz_path, RAW_DATA_DIR)

    # 检查 simple-examples 目录结构
    simple_examples_dir = os.path.join(RAW_DATA_DIR, "simple-examples")
    if os.path.exists(simple_examples_dir):
        # 将 data/ 下的文件移到 raw/ 目录
        data_subdir = os.path.join(simple_examples_dir, "data")
        if os.path.exists(data_subdir):
            import shutil
            for f in os.listdir(data_subdir):
                src = os.path.join(data_subdir, f)
                dst = os.path.join(RAW_DATA_DIR, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  复制: {f} → {RAW_DATA_DIR}")

    # 最终检查
    print("\n最终检查:")
    if check_raw_data():
        print("\n数据准备完成!")
    else:
        print("\n警告: 部分文件缺失，请检查解压结果。")


if __name__ == "__main__":
    main()
