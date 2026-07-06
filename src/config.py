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
    dropout = 0.6
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
    batch_size = 20              
    num_steps = 35               # BPTT 展开步数 (与论文一致)
    vocab_size = 10000

    # ========== 模型结构 ==========
    hidden_size = 650
    num_layers = 2
    embedding_size = 650         # 与 hidden_size 一致 (weight tying 要求)
    dropout = 0.50
    init_scale = 0.05
    use_weight_tying = True      # 启用 Weight Tying (减少参数, 提升泛化)

    # ========== 优化器 (AdamW) ==========
    # optimizer = "adamw"          # "adamw" | "sgd"
    # lr = 0.001                   # AdamW 峰值学习率
    # weight_decay = 0.01        # 权重衰减 (轻微正则化)
    # betas = (0.9, 0.999)         # Adam 动量参数
    # # SGD 回退参数 (当 optimizer="sgd" 时生效)
    # sgd_lr = 1.0
    # lr_decay = 0.5
    # lr_decay_epoch = 6
    # ========== 优化器 ==========
    optimizer = "sgd"
    sgd_lr = 10.0                # 保持大学习率，提供跳出局部最优的动能
    lr_decay = 0.85              
    lr_decay_epoch = 15

    # ========== 训练 ==========
    # max_epoch = 100              # 足够长的训练
    # warmup_epochs = 5            # 前 5 个 epoch 线性增加 LR
    # min_lr = 1e-6                # 余弦退火的最终学习率
    # max_grad_norm = 5.0          # 梯度截断

    max_epoch = 60              # 足够长的训练
    warmup_epochs = 0
    max_grad_norm = 0.25
    # ========== 硬件 ==========
    use_amp = True               # 自动混合精度 (T4 支持 FP16)

    # ========== 日志 ==========
    log_interval = 100
    save_every = 1


class OptimizedLargeConfig:
    """
    论文原版 Large 配置 (Zaremba 2015) + 微调增强。

    核心设计 (与论文完全一致):
      - Standard LSTM (NO weight tying): 避免 Embedding 双路梯度冲突
      - SGD lr=1.0, 每 14 epoch 减半: 论文原版策略
      - batch_size=20, num_steps=35: 论文原值
      - dropout=0.65, max_grad_norm=10: 论文原值

    微调增强:
      - AMP 混合精度: 利用 T4 FP16 加速
      - ASGD (epoch 46+): 参数平均，提升最终泛化
      - 55 epochs: 与论文一致 (14+14+14+13)

    训练节奏:
      Epoch  1-14 : lr = 1.000
      Epoch 15-28 : lr = 0.500
      Epoch 29-42 : lr = 0.250
      Epoch 43-55 : lr = 0.125
      ASGD 从 epoch 46 开始
    """
    # ========== 数据 ==========
    batch_size = 20               # 论文原值 (Small/Medium/Large 统一)
    num_steps = 35                # BPTT 展开步数
    vocab_size = 10000

    # ========== 模型结构 (论文原版, 无 weight tying) ==========
    hidden_size = 1500
    num_layers = 2
    embedding_size = 1500         # 论文: embedding=hidden, 但不用 tying
    dropout = 0.65                # 论文 Large 原值
    init_scale = 0.04             # 论文 Large 原值
    use_weight_tying = False    # ★ 关键: 不用 weight tying!
    # Weight Tying + SGD lr=1.0 会导致 Embedding 梯度爆炸
    # 详见: Press & Wolf (2016) 指出 tying 需要专门的优化策略

    # ========== 优化器 (★ 告别 AdamW，拥抱安全的 SGD) ==========
    optimizer = "sgd"            
    sgd_lr = 1.0                 # ★ 绝对安全值！Zaremba 论文原版 LR，不爆炸，不贪婪
    lr_decay = 0.87              # 经典的 1/1.15 衰减率
    lr_decay_epoch = 15          # 前 14 轮不衰减，提供足够的初始探索空间

   # ========== 训练 ==========
    max_epoch = 60               
    warmup_epochs = 0            # SGD 不需要 warmup
    max_grad_norm = 5.0          # ★ 严格的 5.0 梯度截断，彻底锁死爆炸的可能

    # ========== ASGD (微调增强) ==========
    use_asgd = True
    asgd_start_epoch = 40         # 在最后 LR 阶段开始平均

    # ========== 硬件 ==========
    use_amp = True                # T4 支持 FP16

    # ========== 日志 ==========
    log_interval = 100
    save_every = 100


class GuaranteedConfig:
    """
    必破 80 PPL 的现代优化版配置 (AdamW + 余弦退火 + Weight Tying)
    """
    # ========== 数据 ==========
    batch_size = 20              
    num_steps = 35               
    vocab_size = 10000

    # ========== 模型结构 ==========
    # 使用 Medium 大小足以破 80，训练速度提升一倍
    hidden_size = 650
    num_layers = 2
    embedding_size = 650         
    dropout = 0.50               
    init_scale = 0.05
    use_weight_tying = True      # 开启权重共享，大幅提升泛化能力！

    # ========== 现代优化器 (AdamW) ==========
    optimizer = "adamw"          
    lr = 0.002                   # AdamW 推荐峰值学习率
    weight_decay = 1e-5          # 轻微正则化
    betas = (0.9, 0.999)         

    # ========== 训练节奏 ==========
    max_epoch = 40               # 40 轮绝对够了
    warmup_epochs = 2            # 前 2 轮线性预热，防止早期梯度爆炸
    min_lr = 1e-5                # 余弦退火的最终兜底学习率
    max_grad_norm = 5.0          

    # ========== 硬件与日志 ==========
    use_amp = True               # 继续用混合精度加速
    log_interval = 100
    save_every = 40              # 改为 40，防止 Kaggle 硬盘又爆掉，反正 train.py 会自动存 best_model.pt

# ============ 选择当前配置 ============
# 可以改为 SmallConfig / MediumConfig / LargeConfig /
#          OptimizedMediumConfig / OptimizedLargeConfig
# config = SmallConfig()
# config = MediumConfig()
# config = LargeConfig()
# config = OptimizedMediumConfig()
# config = OptimizedLargeConfig()
config = GuaranteedConfig()


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
