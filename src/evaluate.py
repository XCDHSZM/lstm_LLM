"""
评估脚本 —— 计算并在验证集/测试集上输出困惑度 (Perplexity)
可以加载已训练的模型进行单独评估。
"""

import os
import math
import json
import torch
import torch.nn as nn

from config import (
    config,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    device,
    TRAIN_IDS, VALID_IDS, TEST_IDS, VOCAB_FILE,
)
from data_loader import load_processed, preprocess_data, batch_generator
from model import Model


def evaluate_model(model, data, batch_size, num_steps, device="cpu"):
    """
    评估模型并返回 loss 和困惑度。

    Args:
        model: 已加载的 LSTM 语言模型
        data: numpy array of token IDs
        batch_size: batch 大小
        num_steps: BPTT 展开步数
        device: 计算设备

    Returns:
        (loss, perplexity)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for inputs, targets in batch_generator(data, batch_size, num_steps, device):
            # 截断 BPTT
            hidden = model.detach_hidden(hidden)

            # 前向传播
            logits, hidden = model(inputs, hidden)

            # 计算 loss
            logits = logits.reshape(-1, model.vocab_size)
            targets = targets.reshape(-1)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)

    return avg_loss, avg_ppl


def evaluate_from_checkpoint(checkpoint_name: str = "best_model.pt"):
    """
    从保存的检查点加载模型并评估。

    Args:
        checkpoint_name: 检查点文件名 (相对于 MODEL_DIR)
    """
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        print("请先运行 train.py 训练模型。")
        return

    print(f"加载模型: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从检查点获取配置
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
    else:
        # 回退到默认配置
        ckpt_config = {
            "vocab_size": config.vocab_size,
            "embedding_size": config.embedding_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        }

    # 如果检查点保存了词表，使用它
    if "vocab" in checkpoint:
        vocab = checkpoint["vocab"]
        vocab_size = len(vocab)
    else:
        # 尝试从文件加载
        vocab_path = os.path.join(PROCESSED_DATA_DIR, VOCAB_FILE)
        if os.path.exists(vocab_path):
            vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
            vocab_size = len(vocab)
        else:
            vocab_size = ckpt_config.get("vocab_size", config.vocab_size)
            vocab = None

    print(f"检查点信息:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - 训练 PPL: {checkpoint.get('train_ppl', 'N/A')}")
    print(f"  - 验证 PPL: {checkpoint.get('valid_ppl', 'N/A')}")

    # 构建模型
    model = Model(
        vocab_size=vocab_size,
        embedding_size=ckpt_config["embedding_size"],
        hidden_size=ckpt_config["hidden_size"],
        num_layers=ckpt_config["num_layers"],
        dropout=ckpt_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 确保数据已预处理
    _, valid_ids, test_ids, _ = preprocess_data()

    # 评估
    print("\n" + "=" * 60)
    print("评估中...")
    print("=" * 60)

    valid_loss, valid_ppl = evaluate_model(
        model, valid_ids, config.batch_size, config.num_steps, device
    )
    print(f"\n验证集:")
    print(f"  Loss: {valid_loss:.3f}")
    print(f"  Perplexity: {valid_ppl:.1f}")

    test_loss, test_ppl = evaluate_model(
        model, test_ids, config.batch_size, config.num_steps, device
    )
    print(f"\n测试集:")
    print(f"  Loss: {test_loss:.3f}")
    print(f"  Perplexity: {test_ppl:.1f}")

    print(f"\n{'='*60}")
    print(f"总结:")
    print(f"  验证 PPL: {valid_ppl:.1f}")
    print(f"  测试 PPL: {test_ppl:.1f}")
    print(f"{'='*60}")

    return valid_ppl, test_ppl


def evaluate_all_splits():
    """
    在验证集和测试集上都进行评估（不加载检查点，需要先有预处理数据）。
    用于在训练过程中快速手动评估。
    """
    # 确保数据已预处理
    _, valid_ids, test_ids, vocab = preprocess_data()

    # 尝试加载最佳模型
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到 {checkpoint_path}")
        return

    return evaluate_from_checkpoint("best_model.pt")


def quick_evaluate(model, data_name="valid"):
    """
    快速评估函数 —— 可在训练脚本中直接调用。

    Args:
        model: 模型实例
        data_name: "train" | "valid" | "test"

    Returns:
        (loss, perplexity)
    """
    train_ids, valid_ids, test_ids, _ = preprocess_data()

    data_map = {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }

    data = data_map.get(data_name, valid_ids)
    return evaluate_model(model, data, config.batch_size, config.num_steps, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估 LSTM 语言模型")
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt",
        help="检查点文件名 (默认: best_model.pt)"
    )
    args = parser.parse_args()

    evaluate_from_checkpoint(args.checkpoint)
