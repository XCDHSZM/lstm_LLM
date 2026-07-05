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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import config, MODEL_DIR, device
from data_loader import preprocess_data, batch_generator
from model import Model


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


def compute_loss(model, batch, hidden, criterion, use_amp=False):
    """
    计算单个 batch 的损失。

    Args:
        model: LSTM 模型
        batch: (input, target) 元组
        hidden: 隐状态
        criterion: 损失函数
        use_amp: 是否使用 AMP 自动混合精度

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
                device, epoch):
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
        with autocast(enabled=use_amp):
            loss, hidden = compute_loss(model, batch, hidden, criterion, use_amp)

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
        loss, hidden = compute_loss(model, batch, hidden, criterion, use_amp=False)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)

    return avg_loss, avg_ppl


# ============================================================
# 学习率调度器
# ============================================================

def create_scheduler(optimizer, warmup_epochs, max_epoch, min_lr, initial_lr, optimizer_type="adamw"):
    """
    创建学习率调度器。

    - AdamW: 线性 warmup → 余弦退火
    - SGD: StepLR（原始论文方式）
    """
    if optimizer_type == "adamw":
        # 线性 warmup + 余弦退火
        # LambdaLR 的 lambda 函数接收 epoch (0-indexed)，返回乘数因子
        # 实际 LR = initial_lr * lr_lambda(epoch)
        lr_ratio = min_lr / max(initial_lr, 1e-12)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup: 从 initial_lr/warmup_epochs 到 initial_lr
                return (epoch + 1) / max(1, warmup_epochs)
            else:
                # 余弦退火: 从 initial_lr 衰减到 min_lr
                progress = (epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # 不要低于 min_lr / initial_lr 的比例
                return max(lr_ratio, cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # SGD: StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_decay_epoch,
            gamma=config.lr_decay,
        )

    return scheduler


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
    print(f"Weight Tying: {'启用' if config.use_weight_tying else '禁用'}")
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
    model = Model(
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

    # ========== 检查多 GPU ==========
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块 GPU，启用 DataParallel")
        model = nn.DataParallel(model)

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.sgd_lr)

    # 学习率调度器
    initial_lr = config.lr if config.optimizer == "adamw" else config.sgd_lr
    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epoch=config.max_epoch,
        min_lr=config.min_lr,
        initial_lr=initial_lr,
        optimizer_type=config.optimizer,
    )

    # AMP 混合精度 (仅 CUDA 可用)
    use_amp = config.use_amp and device == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("  AMP 混合精度已启用")

    # ========== 训练循环 ==========
    best_valid_ppl = float("inf")
    best_epoch = 0
    train_history = []
    valid_history = []
    lr_history = []

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(1, config.max_epoch + 1):
        epoch_start = time.time()
        cur_lr = get_lr(optimizer)
        lr_history.append(cur_lr)

        # 训练
        train_loss, train_ppl = train_epoch(
            model, train_ids, optimizer, criterion, scaler,
            config.batch_size, config.num_steps,
            config.hidden_size, config.num_layers,
            device, epoch,
        )

        # 验证
        valid_loss, valid_ppl = evaluate(
            model, valid_ids,
            config.batch_size, config.num_steps,
            config.hidden_size, config.num_layers,
            device,
        )

        # 学习率衰减 (在每个 epoch 之后)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 记录历史
        train_history.append(train_ppl)
        valid_history.append(valid_ppl)

        # 打印结果
        improved = "★" if valid_ppl < best_valid_ppl else " "
        print(f"\nEpoch {epoch:3d}/{config.max_epoch} | "
              f"Time: {epoch_time:.1f}s | LR: {cur_lr:.2e} {improved}")
        print(f"  Train  Loss: {train_loss:.3f} | PPL: {train_ppl:.1f}")
        print(f"  Valid  Loss: {valid_loss:.3f} | PPL: {valid_ppl:.1f}")

        # 保存最佳模型
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            best_epoch = epoch

            # 获取底层模型（处理 DataParallel 包装）
            save_model = model.module if hasattr(model, "module") else model
            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
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
            save_model = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

        # 早停检查：如果 PPL 长时间不降
        if epoch - best_epoch > 20 and config.optimizer == "sgd":
            print(f"\n  验证 PPL 已 {epoch - best_epoch} 轮未改善，提前停止。")
            break

    # ========== 最终测试 ==========
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                           map_location=device)

    # 加载到模型（处理 DataParallel）
    save_model = model.module if hasattr(model, "module") else model
    save_model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_ppl = evaluate(
        model, test_ids,
        config.batch_size, config.num_steps,
        config.hidden_size, config.num_layers,
        device,
    )

    print(f"\n{'='*60}")
    print(f"最终结果 (最佳 epoch: {best_epoch}):")
    print(f"  验证 PPL:   {best_valid_ppl:.1f}")
    print(f"  测试 PPL:   {test_ppl:.1f}")
    print(f"  目标 PPL:   < 80")
    status = "✓ 达标!" if test_ppl < 80 else "✗ 未达标，考虑训练更久或调整超参数"
    print(f"  达标判断:   {status}")
    print(f"{'='*60}")

    return model, train_history, valid_history, best_valid_ppl, test_ppl


if __name__ == "__main__":
    train()
