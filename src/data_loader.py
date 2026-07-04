"""
数据加载与预处理模块
- 从原始 PTB 文本构建词表
- 将文本转换为 ID 序列
- 生成 BPTT 训练所需的 batch
"""

import os
import json
import numpy as np
import torch
from collections import Counter

from config import (
    config,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_FILE, VALID_FILE, TEST_FILE,
    TRAIN_IDS, VALID_IDS, TEST_IDS,
    VOCAB_FILE,
)


# 特殊 token
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


def read_raw_file(filename: str) -> list[str]:
    """读取原始 PTB 文本, 返回 token 列表。"""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.extend(line.split())
    return tokens


def build_vocab(tokens: list[str], vocab_size: int) -> dict:
    """
    根据词频构建词典。
    保留 <pad>, <unk>, <sos>, <eos> 四个特殊 token。
    """
    counter = Counter(tokens)
    # 按词频降序排列，取前 vocab_size - 4 个（留给特殊token）
    most_common = counter.most_common(vocab_size - 4)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        SOS_TOKEN: 2,
        EOS_TOKEN: 3,
    }
    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict) -> list[int]:
    """将 token 列表转换为 ID 列表，未知词用 <unk>。"""
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(token, unk_id) for token in tokens]


def replace_rare_words(tokens: list[str], vocab: dict) -> list[str]:
    """将不在词典中的低频词替换为 <unk>。"""
    return [token if token in vocab else UNK_TOKEN for token in tokens]


def save_processed(filepath: str, data):
    """保存处理后的数据。"""
    np.save(filepath, np.array(data, dtype=np.int64))


def load_processed(filepath: str) -> np.ndarray:
    """加载处理后的数据。"""
    return np.load(filepath)


def preprocess_data():
    """
    完整的数据预处理流程：
    读取原始文件 → 构建词典 → 转换为 ID → 保存到 processed 目录。
    如果已存在处理好的文件则跳过。
    """
    train_ids_path = os.path.join(PROCESSED_DATA_DIR, TRAIN_IDS)
    valid_ids_path = os.path.join(PROCESSED_DATA_DIR, VALID_IDS)
    test_ids_path = os.path.join(PROCESSED_DATA_DIR, TEST_IDS)
    vocab_path = os.path.join(PROCESSED_DATA_DIR, VOCAB_FILE)

    # 如果已处理过，直接返回
    if all(os.path.exists(p) for p in [train_ids_path, valid_ids_path, test_ids_path, vocab_path]):
        print("[数据预处理] 已存在处理好的文件，跳过预处理。")
        vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        train_ids = load_processed(train_ids_path)
        valid_ids = load_processed(valid_ids_path)
        test_ids = load_processed(test_ids_path)
        return train_ids, valid_ids, test_ids, vocab

    print("[数据预处理] 开始预处理 PTB 数据...")

    # 读取原始数据
    print("  - 读取训练数据...")
    train_tokens = read_raw_file(TRAIN_FILE)
    print(f"    训练集 token 数: {len(train_tokens)}")

    print("  - 读取验证数据...")
    valid_tokens = read_raw_file(VALID_FILE)
    print(f"    验证集 token 数: {len(valid_tokens)}")

    print("  - 读取测试数据...")
    test_tokens = read_raw_file(TEST_FILE)
    print(f"    测试集 token 数: {len(test_tokens)}")

    # 构建词典（只用训练集）
    print(f"  - 构建词典 (vocab_size={config.vocab_size})...")
    vocab = build_vocab(train_tokens, config.vocab_size)
    print(f"    实际词表大小: {len(vocab)}")

    # 替换低频词为 <unk>
    train_tokens = replace_rare_words(train_tokens, vocab)
    valid_tokens = replace_rare_words(valid_tokens, vocab)
    test_tokens = replace_rare_words(test_tokens, vocab)

    # 转换为 ID
    train_ids = tokens_to_ids(train_tokens, vocab)
    valid_ids = tokens_to_ids(valid_tokens, vocab)
    test_ids = tokens_to_ids(test_tokens, vocab)

    # 保存
    print("  - 保存处理后的数据...")
    save_processed(train_ids_path, train_ids)
    save_processed(valid_ids_path, valid_ids)
    save_processed(test_ids_path, test_ids)
    json.dump(vocab, open(vocab_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[数据预处理] 完成！文件保存在: {PROCESSED_DATA_DIR}")
    print(f"  - 词典大小: {len(vocab)}")
    print(f"  - 训练 tokens: {len(train_ids)}")
    print(f"  - 验证 tokens: {len(valid_ids)}")
    print(f"  - 测试 tokens: {len(test_ids)}")

    return train_ids, valid_ids, test_ids, vocab


def batch_generator(data: np.ndarray, batch_size: int, num_steps: int, device: str = "cpu"):
    """
    生成 BPTT 训练所需的 batch。

    数据被重塑为 [batch_size, -1] 的形状，然后按 num_steps 切片。
    每个 batch 返回 (input, target)，其中 target 是 input 右移一位。

    Yields:
        (input, target): 形状均为 [batch_size, num_steps]
    """
    data_len = len(data)
    # 截断为 batch_size 的整数倍
    batch_len = data_len // batch_size
    data = data[:batch_size * batch_len]
    data = data.reshape(batch_size, batch_len)  # [batch_size, batch_len]

    # 按 num_steps 滑动窗口
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


def get_batch(data: np.ndarray, batch_size: int, num_steps: int, device: str = "cpu"):
    """返回所有 batch 的列表（非生成器），方便 debug。"""
    return list(batch_generator(data, batch_size, num_steps, device))


if __name__ == "__main__":
    # 测试预处理
    train_ids, valid_ids, test_ids, vocab = preprocess_data()
    print(f"\n词表前 20 个词: {list(vocab.keys())[:20]}")

    # 测试 batch 生成
    batches = get_batch(train_ids, batch_size=config.batch_size,
                        num_steps=config.num_steps, device="cpu")
    print(f"\nBatch 数量: {len(batches)}")
    if batches:
        x, y = batches[0]
        print(f"第一个 batch: x.shape={x.shape}, y.shape={y.shape}")
