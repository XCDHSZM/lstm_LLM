"""
训练脚本 —— BPTT 训练循环
包含:
  - 截断 BPTT (Truncated Backpropagation Through Time)
  - 梯度截断 (Gradient Clipping)
  - 学习率衰减 (Learning Rate Decay)
  - 模型检查点保存
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import config, MODEL_DIR, device
from data_loader import preprocess_data, batch_generator
from model import Model


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

    # logits: [batch_size, seq_len, vocab_size] → [batch_size * seq_len, vocab_size]
    # targets: [batch_size, seq_len] → [batch_size * seq_len]
    logits = logits.reshape(-1, model.vocab_size)
    targets = targets.reshape(-1)

    # 交叉熵损失
    loss = criterion(logits, targets)

    return loss, hidden


def train_epoch(model, data, optimizer, criterion, batch_size, num_steps, device, epoch):
    """
    训练一个 epoch。

    Returns:
        avg_loss: 平均 loss
        avg_ppl: 平均困惑度
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size, device)

    # 生成 batch
    batches = list(batch_generator(data, batch_size, num_steps, device))
    pbar = tqdm(batches, desc=f"Epoch {epoch}", unit="batch")

    for i, batch in enumerate(pbar):
        # 截断 BPTT: 将隐状态从计算图中分离
        hidden = model.detach_hidden(hidden)

        # 梯度清零
        optimizer.zero_grad()

        # 计算 loss
        loss, hidden = compute_loss(model, batch, hidden, criterion)

        # 反向传播
        loss.backward()

        # 梯度截断
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # 更新参数
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        if (i + 1) % config.log_interval == 0:
            cur_loss = total_loss / num_batches
            cur_ppl = math.exp(cur_loss)
            pbar.set_postfix({"loss": f"{cur_loss:.3f}", "ppl": f"{cur_ppl:.1f}"})

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)

    return avg_loss, avg_ppl


def evaluate(model, data, batch_size, num_steps, device):
    """
    在数据集上评估模型，返回 loss 和困惑度。
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    hidden = model.init_hidden(batch_size, device)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in batch_generator(data, batch_size, num_steps, device):
            hidden = model.detach_hidden(hidden)
            loss, hidden = compute_loss(model, batch, hidden, criterion)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(avg_loss)

    return avg_loss, avg_ppl


def get_lr(optimizer):
    """获取当前学习率。"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 1.0


def train():
    """
    完整的训练流程。
    """
    print("=" * 60)
    print("LSTM 神经网络语言模型 - PyTorch 实现")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"配置: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"embed={config.embedding_size}, dropout={config.dropout}")
    print(f"训练参数: batch={config.batch_size}, steps={config.num_steps}, "
          f"epochs={config.max_epoch}, lr={config.lr}")

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
    print(f"\n模型参数量: {total_params:,}")

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.CrossEntropyLoss()

    # 使用 SGD（与原始论文一致）
    # 或者可以使用 Adam 获得更快的收敛
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    # 学习率调度器：每个 lr_decay_epoch 衰减一次
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_epoch,
        gamma=config.lr_decay,
    )

    # ========== 训练循环 ==========
    best_valid_ppl = float("inf")
    train_history = []
    valid_history = []

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(1, config.max_epoch + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_ppl = train_epoch(
            model, train_ids, optimizer, criterion,
            config.batch_size, config.num_steps, device, epoch,
        )

        # 学习率衰减
        scheduler.step()
        cur_lr = get_lr(optimizer)

        # 验证
        valid_loss, valid_ppl = evaluate(
            model, valid_ids,
            config.batch_size, config.num_steps, device,
        )

        epoch_time = time.time() - epoch_start

        # 记录历史
        train_history.append(train_ppl)
        valid_history.append(valid_ppl)

        # 打印结果
        print(f"\nEpoch {epoch:2d}/{config.max_epoch} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {cur_lr:.4f}")
        print(f"  Train  Loss: {train_loss:.3f} | PPL: {train_ppl:.1f}")
        print(f"  Valid  Loss: {valid_loss:.3f} | PPL: {valid_ppl:.1f}")

        # 保存最佳模型
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
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
            }, ckpt_path)

    # ========== 最终测试 ==========
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    print("=" * 60)

    best_ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"))
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_ppl = evaluate(
        model, test_ids,
        config.batch_size, config.num_steps, device,
    )

    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  最佳验证 PPL: {best_valid_ppl:.1f}")
    print(f"  测试 PPL:     {test_ppl:.1f}")
    print(f"{'='*60}")

    return model, train_history, valid_history, best_valid_ppl, test_ppl


if __name__ == "__main__":
    train()
