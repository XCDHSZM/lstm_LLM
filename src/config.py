"""
配置参数 —— 基于 Zaremba et al. "Recurrent Neural Network Regularization" (2015)
提供了 Small / Medium / Large / Optimized 多种配置。
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
    # 优化器相关
    optimizer = "sgd"
    use_weight_tying = False
    use_amp = False


class MediumConfig:
    """中等模型配置 —— 较好效果（论文复现）"""
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
    # 优化器相关
    optimizer = "sgd"
    use_weight_tying = False
    use_amp = False


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
    # 优化器相关
    optimizer = "sgd"
    use_weight_tying = False
    use_amp = False


class OptimizedMediumConfig:
    """
    优化版中等配置 — 目标验证 PPL < 80。

    关键改进:
      - AdamW 优化器：比 SGD 稳定得多，收敛更快
      - Weight Tying：共享 Embedding 和输出层权重，减少过拟合
      - 线性 warmup + 余弦退火：稳定训练初期，后期精细收敛
      - 更大的 batch_size：利用 T4 16GB 显存，梯度估计更准确
      - AMP 混合精度：T4 支持 FP16，训练速度提升 ~1.5-2x
      - 更长训练：100 epochs 确保充分收敛
    """
    # ========== 数据 ==========
    batch_size = 64              # 增大 batch 利用 GPU 显存 (T4 16GB 绰绰有余)
    num_steps = 35               # BPTT 展开步数 (与论文一致)
    vocab_size = 10000

    # ========== 模型结构 ==========
    hidden_size = 650
    num_layers = 2
    embedding_size = 650         # 与 hidden_size 一致 (weight tying 要求)
    dropout = 0.5
    init_scale = 0.05
    use_weight_tying = True      # 启用 Weight Tying (减少参数, 提升泛化)

    # ========== 优化器 (AdamW) ==========
    optimizer = "adamw"          # "adamw" | "sgd"
    lr = 0.001                   # AdamW 峰值学习率
    weight_decay = 1e-5          # 权重衰减 (轻微正则化)
    betas = (0.9, 0.999)         # Adam 动量参数
    # SGD 回退参数 (当 optimizer="sgd" 时生效)
    sgd_lr = 1.0
    lr_decay = 0.5
    lr_decay_epoch = 6

    # ========== 训练 ==========
    max_epoch = 100              # 足够长的训练
    warmup_epochs = 5            # 前 5 个 epoch 线性增加 LR
    min_lr = 1e-6                # 余弦退火的最终学习率
    max_grad_norm = 5.0          # 梯度截断

    # ========== 硬件 ==========
    use_amp = True               # 自动混合精度 (T4 支持 FP16)

    # ========== 日志 ==========
    log_interval = 100
    save_every = 1


# ============ 选择当前配置 ============
# 可以改为 SmallConfig / MediumConfig / LargeConfig / OptimizedMediumConfig
# config = SmallConfig()
# config = MediumConfig()
# config = LargeConfig()
config = OptimizedMediumConfig()


# ============ 训练通用设置 ============
device = "cuda"  # 或 "cpu"

# ============ 数据文件 ============
TRAIN_FILE = "ptb.train.txt"
VALID_FILE = "ptb.valid.txt"
TEST_FILE = "ptb.test.txt"

# 预处理后的文件
TRAIN_IDS = "ptb.train.ids.npy"
VALID_IDS = "ptb.valid.ids.npy"
TEST_IDS = "ptb.test.ids.npy"
VOCAB_FILE = "ptb.vocab.json"
