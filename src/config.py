"""
配置参数 —— 基于 Zaremba et al. "Recurrent Neural Network Regularization" (2015)
提供了 Small / Medium / Large 三种配置
"""

import os

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


class SmallConfig:
    """小模型配置 —— 快速实验"""
    batch_size = 20
    num_steps = 20          # BPTT 展开步数
    hidden_size = 200       # LSTM 隐层大小
    num_layers = 2          # LSTM 层数
    dropout = 0.5           # Dropout 概率
    init_scale = 0.1        # 权重初始化范围
    max_grad_norm = 5.0     # 梯度截断阈值
    lr = 1.0                # 初始学习率
    lr_decay = 0.5          # 学习率衰减因子
    max_epoch = 13          # 最大训练 epoch 数
    lr_decay_epoch = 4      # 每多少个 epoch 学习率衰减
    vocab_size = 10000      # 词典大小
    embedding_size = 200    # 词向量维度
    log_interval = 50       # 每 N 步打印一次训练 loss
    save_every = 1          # 每 N 个 epoch 保存一次模型


class MediumConfig:
    """中等模型配置 —— 较好效果"""
    batch_size = 20
    num_steps = 35
    hidden_size = 650
    num_layers = 2
    dropout = 0.5
    init_scale = 0.05
    max_grad_norm = 5.0
    lr = 1.0
    lr_decay = 0.8
    max_epoch = 50
    lr_decay_epoch = 6
    vocab_size = 10000
    embedding_size = 650
    log_interval = 100
    save_every = 1


class LargeConfig:
    """大模型配置 —— 最佳效果（需要 GPU）"""
    batch_size = 20
    num_steps = 35
    hidden_size = 1500
    num_layers = 2
    dropout = 0.65
    init_scale = 0.04
    max_grad_norm = 10.0
    lr = 1.0
    lr_decay = 1.0 / 1.15
    max_epoch = 55
    lr_decay_epoch = 14
    vocab_size = 10000
    embedding_size = 1500
    log_interval = 200
    save_every = 1


# ============ 选择当前配置 ============
# 可以改为 MediumConfig / LargeConfig
# config = SmallConfig()
config = MediumConfig()
# config = LargeConfig()


# ============ 训练相关 ============
device = "cuda"  # 或 "cpu"
save_every = 1   # 每 N 个 epoch 保存一次模型
log_interval = 100  # 每 N 步打印一次训练 loss

# ============ 数据文件 ============
TRAIN_FILE = "ptb.train.txt"
VALID_FILE = "ptb.valid.txt"
TEST_FILE = "ptb.test.txt"

# 预处理后的文件
TRAIN_IDS = "ptb.train.ids.npy"
VALID_IDS = "ptb.valid.ids.npy"
TEST_IDS = "ptb.test.ids.npy"
VOCAB_FILE = "ptb.vocab.json"
