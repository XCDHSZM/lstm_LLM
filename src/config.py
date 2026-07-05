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


class OptimizedLargeConfig:
    """
    优化版大模型配置 — 目标测试 PPL < 80。

    设计思路:
      - Large 模型 (hidden=1500) 是 Zaremba 论文中唯一能达到 <80 PPL 的配置
      - 回归 SGD + 手动 LR 减半 (论文原版策略，比 AdamW 在 LSTM LM 上更优)
      - Weight Tying 减少参数量 (66M → 51M)，同时作为正则化
      - 线性 warmup 防止训练初期震荡
      - ASGD (Averaged SGD) 收尾：训练最后阶段对参数取平均，稳定收敛
      - AMP 混合精度加速 (T4 支持 FP16)

    训练节奏 (参考 Zaremba 2015):
      Epoch   1-2 : warmup (lr: 0.1 → 1.0)
      Epoch  3-16 : lr = 1.0   (14 epochs)
      Epoch 17-30 : lr = 0.5   (14 epochs, 减半)
      Epoch 31-44 : lr = 0.25  (14 epochs, 减半)
      Epoch 45-58 : lr = 0.125 (14 epochs, 减半)
      Epoch 59-70 : lr = 0.0625(12 epochs, 精细收敛)
      ASGD 从 epoch 50 开始
    """
    # ========== 数据 ==========
    batch_size = 32              # Large 模型显存占用更大，batch 不宜过高
    num_steps = 35               # BPTT 展开步数 (与论文一致)
    vocab_size = 10000

    # ========== 模型结构 ==========
    hidden_size = 1500           # Large: 1500 (论文 Large 配置)
    num_layers = 2
    embedding_size = 1500        # 与 hidden_size 一致 (weight tying 要求)
    dropout = 0.65               # Large 用 0.65 (论文原值)
    init_scale = 0.04            # Large 用 0.04 (论文原值)
    use_weight_tying = True

    # ========== 优化器 (SGD) ==========
    optimizer = "sgd"            # SGD 在 LSTM LM 上比 AdamW 效果更好
    sgd_lr = 1.0                 # 初始学习率 (论文原值)
    lr_decay = 0.5               # 每次衰减 ×0.5 (论文原值: halve)
    lr_decay_epoch = 14          # 每 14 个 epoch 衰减一次
    # AdamW 回退参数
    lr = 0.001
    weight_decay = 1e-5
    betas = (0.9, 0.999)

    # ========== 训练 ==========
    max_epoch = 70               # 比论文 55 多 15 epoch 以保证充分收敛
    warmup_epochs = 2            # 前 2 epoch 从 0.1→1.0 线性 warmup
    warmup_start_lr = 0.1        # warmup 起始学习率
    min_lr = 1e-6
    max_grad_norm = 10.0         # Large 用 10 (论文原值)

    # ========== ASGD ==========
    use_asgd = True              # 启用 Averaged SGD
    asgd_start_epoch = 50        # 从第 50 epoch 开始对参数取平均

    # ========== 硬件 ==========
    use_amp = True               # AMP 混合精度 (T4 支持 FP16)

    # ========== 日志 ==========
    log_interval = 100
    save_every = 1


# ============ 选择当前配置 ============
# 可以改为 SmallConfig / MediumConfig / LargeConfig /
#          OptimizedMediumConfig / OptimizedLargeConfig
# config = SmallConfig()
# config = MediumConfig()
# config = LargeConfig()
# config = OptimizedMediumConfig()
config = OptimizedLargeConfig()


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
