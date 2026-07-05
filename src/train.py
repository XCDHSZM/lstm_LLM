"""
训练脚本 —— BPTT 训练循环（优化版）

改进点:
  - AdamW 优化器 (替代 SGD): 更稳定, 收敛更快
  - 线性 warmup + 余弦退火: 训练初期稳定 + 后期精细收敛
  - AMP 混合精度: T4 GPU 支持 FP16, 提速 ~1.5-2x
  - Weight Tying: 共享 Embedding 和输出层权重, 减少过拟合
  - 更大的 batch size: 利用 T4 16GB 显存, 梯度估计更准确
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# AMP 导入兼容不同 PyTorch 版本
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None

from config import config, MODEL_DIR, device
from data_loader import preprocess_data, batch_generator
from model import LSTMLanguageModel, LSTMLanguageModelTied


# ============================================================
# 工具函数
# ============================================================

def get_model_attribute(model, attr_name):
    """
    获取模型属性，兼容 nn.DataParallel 包装。

    直接访问 model.attr_name，如果不存在则尝试 model.module.attr_name。
    """
    if hasattr(model, attr_name):
        return getattr(model, attr_name)
    elif hasattr(model, "module"):
        return getattr(model.module, attr_name)
    raise AttributeError(f"模型没有 '{attr_name}' 属性")


def compute_loss(model, batch, hidden, criterion):
    """
    计算单个 batch 的损失。

    Args:
        model: LSTM 模型
        batch: (input, target) 元组
        hidden: 隐状态
        criterion: 损失函数

    Returns:
        loss, hidden (detached)
    """
    inputs, targets = batch

    # 前向传播
    logits, hidden = model(inputs, hidden)

    # 获取 vocab_size（兼容 weight tying 模型）
    vocab_size = logits.size(-1)

    # logits: [batch_size, seq_len, vocab_size] → [batch_size * seq_len, vocab_size]
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # 交叉熵损失
    loss = criterion(logits, targets)

    return loss, hidden


def detach_hidden(hidden):
    """截断梯度传播 —— 对 BPTT 至关重要。"""
    if hidden is None:
        return None
    h, c = hidden
    return (h.detach(), c.detach())


def init_hidden(batch_size, hidden_size, num_layers, device):
    """初始化零隐状态（不依赖模型实例，兼容 DataParallel）。"""
    h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    return (h, c)


def get_lr(optimizer):
    """获取当前学习率。"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


# ============================================================
# 训练 & 评估
# ============================================================

def train_epoch(model, data, optimizer, criterion, scaler,
                batch_size, num_steps, hidden_size, num_layers,
                device, epoch, asgd=None):
    """
    训练一个 epoch。

    Returns:
        avg_loss: 平均 loss
        avg_ppl: 平均困惑度
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    hidden = init_hidden(batch_size, hidden_size, num_layers, device)
    use_amp = scaler is not None

    # 生成 batch
    batches = list(batch_generator(data, batch_size, num_steps, device))
    pbar = tqdm(batches, desc=f"Epoch {epoch:3d}", unit="batch")

    for i, batch in enumerate(pbar):
        # 截断 BPTT
        hidden = detach_hidden(hidden)

        # 梯度清零
        optimizer.zero_grad()

        # AMP 前向传播
        with autocast(AMP_DEVICE, enabled=use_amp):
            loss, hidden = compute_loss(model, batch, hidden, criterion)

        # 反向传播（AMP 自动缩放）
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        # ASGD: 每次 step 后更新参数平均
        if asgd is not None:
            asgd.update()

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        if (i + 1) % config.log_interval == 0:
            cur_loss = total_loss / num_batches
            cur_ppl = math.exp(min(cur_loss, 10))  # 防止 loss 爆炸导致 ppl 溢出
            cur_lr = get_lr(optimizer)
            pbar.set_postfix({
                "loss": f"{cur_loss:.3f}",
                "ppl": f"{cur_ppl:.1f}",
                "lr": f"{cur_lr:.2e}",
            })

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(min(avg_loss, 10))

    return avg_loss, avg_ppl


@torch.no_grad()
def evaluate(model, data, batch_size, num_steps, hidden_size, num_layers, device):
    """
    在数据集上评估模型，返回 loss 和困惑度。
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    hidden = init_hidden(batch_size, hidden_size, num_layers, device)
    criterion = nn.CrossEntropyLoss()

    for batch in batch_generator(data, batch_size, num_steps, device):
        hidden = detach_hidden(hidden)
        loss, hidden = compute_loss(model, batch, hidden, criterion)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)

    return avg_loss, avg_ppl


# ============================================================
# 学习率调度器
# ============================================================

def create_scheduler(optimizer, warmup_epochs, max_epoch, min_lr, initial_lr):
    """
    创建 AdamW 学习率调度器: 线性 warmup → 余弦退火。

    SGD 不使用调度器，而是通过 set_epoch_lr() 在每个 epoch 手动设置 LR。
    """
    lr_ratio = min_lr / max(initial_lr, 1e-12)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_epoch_lr(optimizer, epoch, warmup_epochs, warmup_start_lr, sgd_lr,
                 lr_decay, lr_decay_epoch):
    """
    SGD 手动 LR 控制: 线性 warmup → StepLR halving。

    在每个 epoch 开始前调用，精确控制学习率。
    """
    if epoch <= warmup_epochs:
        # 线性 warmup: warmup_start_lr → sgd_lr
        progress = epoch / max(1, warmup_epochs)
        lr = warmup_start_lr + (sgd_lr - warmup_start_lr) * progress
    else:
        # StepLR: 每 lr_decay_epoch 减半
        effective_epoch = epoch - warmup_epochs - 1  # warmup 结束后的第一个完整 epoch 开始 decay
        num_decays = effective_epoch // lr_decay_epoch
        lr = sgd_lr * (lr_decay ** max(0, num_decays))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ============================================================
# ASGD (Averaged SGD) — 参数平均
# ============================================================

class ASGDPolyak:
    """
    Polyak 平均 (参数指数移动平均)。

    在训练后期维护一个 shadow copy 的参数，用 EMA 平滑：
        shadow = decay * shadow + (1 - decay) * current_params

    评估时使用 shadow 参数代替原始参数。
    """

    def __init__(self, model, decay=0.997):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._init_shadow()
        self.num_updates = 0

    def _init_shadow(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone().detach()

    def update(self):
        """在每次 optimizer.step() 后调用。"""
        self.num_updates += 1
        # 用 bias-corrected decay
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param.data

    def apply_shadow(self):
        """将 shadow 参数复制到模型中（评估前调用）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {
            "shadow": self.shadow,
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.num_updates = state_dict["num_updates"]
        self.decay = state_dict["decay"]


# ============================================================
# 主训练函数
# ============================================================

def train():
    """
    完整的训练流程（优化版）。
    """
    print("=" * 60)
    print("LSTM 神经网络语言模型 - PyTorch 实现 (优化版)")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"配置: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"embed={config.embedding_size}, dropout={config.dropout}")
    print(f"优化器: {config.optimizer.upper()}")
    print(f"Weight Tying: {'启用' if getattr(config, 'use_weight_tying', False) else '禁用'}")
    print(f"AMP 混合精度: {'启用' if config.use_amp else '禁用'}")
    print(f"训练参数: batch={config.batch_size}, steps={config.num_steps}, "
          f"epochs={config.max_epoch}")
    if config.optimizer == "adamw":
        print(f"  LR={config.lr}, warmup={config.warmup_epochs} epochs, "
              f"min_lr={config.min_lr}, weight_decay={config.weight_decay}")
    else:
        print(f"  SGD LR={config.sgd_lr}, decay={config.lr_decay} "
              f"every {config.lr_decay_epoch} epochs")

    # ========== 数据预处理 ==========
    train_ids, valid_ids, test_ids, vocab = preprocess_data()

    print(f"\n训练 tokens: {len(train_ids):,}")
    print(f"验证 tokens: {len(valid_ids):,}")
    print(f"测试 tokens: {len(test_ids):,}")
    print(f"词表大小: {len(vocab)}")

    # ========== 构建模型 ==========
    # 根据配置选择模型类型
    if getattr(config, "use_weight_tying", False):
        ModelClass = LSTMLanguageModelTied
        assert config.embedding_size == config.hidden_size, \
            "Weight tying requires embedding_size == hidden_size"
    else:
        ModelClass = LSTMLanguageModel

    model = ModelClass(
        vocab_size=len(vocab),
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        init_scale=config.init_scale,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} (trainable: {trainable_params:,})")

    # ========== GPU 信息 ==========
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块 GPU")
        print("  注意: DataParallel 与 LSTM 隐状态传递不兼容，将使用单 GPU 训练")
        print(f"  当前 batch_size={config.batch_size} 足以充分利用单块 T4 显存")
        print(f"  如需利用多 GPU，建议使用 DistributedDataParallel (需要 torchrun)")

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        scheduler = create_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epoch=config.max_epoch,
            min_lr=config.min_lr,
            initial_lr=config.lr,
        )
        use_scheduler = True
    else:
        # SGD: 手动控制 LR，从 warmup_start_lr 开始
        warmup_start = getattr(config, "warmup_start_lr", config.sgd_lr * 0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=warmup_start)
        scheduler = None
        use_scheduler = False

    # AMP 混合精度 (仅 CUDA 可用)
    use_amp = config.use_amp and device == "cuda"
    if use_amp:
        scaler = GradScaler(AMP_DEVICE) if AMP_DEVICE else GradScaler()
        print("  AMP 混合精度已启用")
    else:
        scaler = None

    # ========== 训练循环 ==========
    best_valid_ppl = float("inf")
    best_epoch = 0
    train_history = []
    valid_history = []
    lr_history = []
    asgd = None
    use_asgd = getattr(config, "use_asgd", False)
    asgd_start = getattr(config, "asgd_start_epoch", config.max_epoch + 1)

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    if use_asgd:
        print(f"ASGD: 从 epoch {asgd_start} 开始对参数取平均")

    for epoch in range(1, config.max_epoch + 1):
        epoch_start = time.time()

        # SGD: 手动设置 epoch LR
        if not use_scheduler:
            cur_lr = set_epoch_lr(
                optimizer, epoch,
                warmup_epochs=config.warmup_epochs,
                warmup_start_lr=getattr(config, "warmup_start_lr", config.sgd_lr * 0.1),
                sgd_lr=config.sgd_lr,
                lr_decay=config.lr_decay,
                lr_decay_epoch=config.lr_decay_epoch,
            )
        else:
            cur_lr = get_lr(optimizer)

        lr_history.append(cur_lr)

        # ASGD: 到达启动 epoch 时创建 shadow 参数
        if use_asgd and epoch == asgd_start:
            asgd = ASGDPolyak(model)
            print(f"\n  [ASGD] 开始参数平均 (epoch {epoch})")

        # 训练
        train_loss, train_ppl = train_epoch(
            model, train_ids, optimizer, criterion, scaler,
            config.batch_size, config.num_steps,
            config.hidden_size, config.num_layers,
            device, epoch, asgd=asgd,
        )

        # 验证（用原始参数，不用 shadow — shadow 仅用于最终测试）
        valid_loss, valid_ppl = evaluate(
            model, valid_ids,
            config.batch_size, config.num_steps,
            config.hidden_size, config.num_layers,
            device,
        )

        # 学习率衰减 (AdamW: 在每个 epoch 之后; SGD: 手动设置)
        if use_scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # 记录历史
        train_history.append(train_ppl)
        valid_history.append(valid_ppl)

        # 打印结果
        improved = "★" if valid_ppl < best_valid_ppl else " "
        asgd_tag = " [ASGD]" if asgd is not None else ""
        print(f"\nEpoch {epoch:3d}/{config.max_epoch} | "
              f"Time: {epoch_time:.1f}s | LR: {cur_lr:.2e}{asgd_tag} {improved}")
        print(f"  Train  Loss: {train_loss:.3f} | PPL: {train_ppl:.1f}")
        print(f"  Valid  Loss: {valid_loss:.3f} | PPL: {valid_ppl:.1f}")

        # 保存最佳模型（用原始参数）
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            best_epoch = epoch
            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_ppl": valid_ppl,
                "train_ppl": train_ppl,
                "config": {
                    "vocab_size": len(vocab),
                    "embedding_size": config.embedding_size,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout,
                    "use_weight_tying": config.use_weight_tying,
                },
                "vocab": vocab,
            }, save_path)
            print(f"  *** 保存最佳模型 (PPL={valid_ppl:.1f}) → {save_path}")

        # 定期保存 checkpoint
        if epoch % config.save_every == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "asgd_state_dict": asgd.state_dict() if asgd else None,
            }, ckpt_path)

        # 早停：SGD 下如果 PPL 长时间不降
        if epoch - best_epoch > 20 and config.optimizer == "sgd" and epoch > 50:
            print(f"\n  验证 PPL 已 {epoch - best_epoch} 轮未改善，提前停止。")
            break

    # ========== 最终测试 ==========
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                           map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # 测试 1: 用原始最佳参数
    test_loss, test_ppl = evaluate(
        model, test_ids,
        config.batch_size, config.num_steps,
        config.hidden_size, config.num_layers,
        device,
    )

    # 测试 2: 用 ASGD 平均参数 (如果有)
    test_ppl_asgd = None
    if asgd is not None:
        asgd.apply_shadow()
        test_loss_asgd, test_ppl_asgd = evaluate(
            model, test_ids,
            config.batch_size, config.num_steps,
            config.hidden_size, config.num_layers,
            device,
        )
        # 恢复原始参数
        model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\n{'='*60}")
    print(f"最终结果 (最佳 epoch: {best_epoch}):")
    print(f"  验证 PPL:        {best_valid_ppl:.1f}")
    print(f"  测试 PPL (原始):  {test_ppl:.1f}")
    if test_ppl_asgd is not None:
        print(f"  测试 PPL (ASGD):  {test_ppl_asgd:.1f}")
    final_ppl = test_ppl_asgd if test_ppl_asgd is not None else test_ppl
    print(f"  目标 PPL:        < 80")
    status = "✓ 达标!" if final_ppl < 80 else "✗ 未达标"
    print(f"  达标判断:        {status}")
    print(f"{'='*60}")

    return model, train_history, valid_history, best_valid_ppl, final_ppl


if __name__ == "__main__":
    train()
