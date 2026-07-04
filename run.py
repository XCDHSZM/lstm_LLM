"""
运行入口脚本 —— 一键执行：数据下载 → 预处理 → 训练 → 评估
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LSTM 神经网络语言模型 (PyTorch 实现)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["download", "preprocess", "train", "evaluate", "all"],
        help="运行模式: download(下载数据) / preprocess(预处理) / train(训练) / evaluate(评估) / all(完整流程)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt",
        help="评估时加载的检查点文件名 (默认: best_model.pt)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="强制使用 CPU"
    )

    args = parser.parse_args()

    if args.cpu:
        import config
        config.device = "cpu"
        print("强制使用 CPU")

    if args.mode == "download":
        import download_data
        download_data.main()

    elif args.mode == "preprocess":
        from data_loader import preprocess_data
        preprocess_data()

    elif args.mode == "train":
        from train import train
        train()

    elif args.mode == "evaluate":
        from evaluate import evaluate_from_checkpoint
        evaluate_from_checkpoint(args.checkpoint)

    elif args.mode == "all":
        print("=" * 60)
        print("完整流程: 数据预处理 → 训练 → 评估")
        print("=" * 60)

        # 1. 检查数据（如果不存在则尝试下载）
        from config import RAW_DATA_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE
        expected = [TRAIN_FILE, VALID_FILE, TEST_FILE]
        all_exist = all(
            os.path.exists(os.path.join(RAW_DATA_DIR, f)) for f in expected
        )
        if not all_exist:
            print("\n⚠ 缺少原始数据，尝试下载...")
            import download_data
            download_data.main()
            # 再次检查
            all_exist = all(
                os.path.exists(os.path.join(RAW_DATA_DIR, f)) for f in expected
            )
            if not all_exist:
                print("\n❌ 数据下载失败，请手动下载后重试。")
                print("  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz")
                return

        # 2. 预处理 & 训练
        from train import train
        train()

        # 3. 评估
        from evaluate import evaluate_from_checkpoint
        evaluate_from_checkpoint("best_model.pt")


if __name__ == "__main__":
    main()
