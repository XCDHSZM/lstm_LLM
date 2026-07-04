"""
LSTM 神经网络语言模型
基于 Zaremba et al. (2015) 的架构：
  Embedding → Dropout → LSTM (多层) → Dropout → Linear → Softmax

权重初始化使用均匀分布 [-init_scale, init_scale]。
LSTM 的 forget gate bias 初始化为 1.0（推荐做法）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    """
    LSTM 语言模型。

    Args:
        vocab_size: 词典大小
        embedding_size: 词嵌入维度
        hidden_size: LSTM 隐层大小
        num_layers: LSTM 层数
        dropout: Dropout 概率（应用于 Embedding 后和 LSTM 输出后）
        init_scale: 权重初始化范围 [-init_scale, init_scale]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Dropout（应用于 embedding 之后）
        self.embed_dropout = nn.Dropout(dropout)

        # LSTM 层
        # 使用 PyTorch 内置的 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch 的 dropout 仅在多层时生效
            batch_first=True,
        )

        # LSTM 输出后的 Dropout
        self.output_dropout = nn.Dropout(dropout)

        # 输出投影层 (hidden_size → vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

        # 权重初始化
        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float):
        """均匀分布初始化所有参数。"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -init_scale, init_scale)
            elif "bias" in name:
                nn.init.zeros_(param)
                # LSTM forget gate bias 初始化为 1.0
                if "lstm" in name and "bias_hh" in name:
                    # bias_hh shape: [4 * hidden_size]
                    # forget gate 的 bias 在后 1/4 的位置
                    n = param.size(0) // 4
                    param.data[n:2 * n].fill_(1.0)  # LSTM forget gate bias = 1

    def forward(
        self,
        input_ids: torch.Tensor,       # [batch_size, seq_len]
        hidden: tuple | None = None,   # 可选的初始隐状态
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播。

        Args:
            input_ids: [batch_size, seq_len] 输入 token IDs
            hidden: 可选的 (h_0, c_0) 元组

        Returns:
            logits: [batch_size, seq_len, vocab_size] 未归一化的预测
            hidden: 最终的 (h_n, c_n) 元组，可用于状态传递
        """
        batch_size, seq_len = input_ids.shape

        # Embedding: [batch_size, seq_len] → [batch_size, seq_len, embedding_size]
        emb = self.embedding(input_ids)
        emb = self.embed_dropout(emb)

        # LSTM: [batch_size, seq_len, embedding_size] → [batch_size, seq_len, hidden_size]
        lstm_out, hidden = self.lstm(emb, hidden)

        # Dropout
        lstm_out = self.output_dropout(lstm_out)

        # 投影到词典大小: [batch_size, seq_len, hidden_size] → [batch_size, seq_len, vocab_size]
        logits = self.linear(lstm_out)

        return logits, hidden

    def detach_hidden(self, hidden: tuple) -> tuple:
        """
        截断梯度传播 —— 对 BPTT 至关重要。
        将隐状态从计算图中分离，防止梯度在 batch 之间反向传播。
        """
        if hidden is None:
            return None
        h, c = hidden
        return (h.detach(), c.detach())

    def init_hidden(self, batch_size: int, device: str = "cpu") -> tuple:
        """初始化零隐状态。"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class LSTMLanguageModelTied(nn.Module):
    """
    使用 weight tying 的 LSTM 语言模型变体。

    将 Embedding 层的权重和输出 Linear 层的权重共享，
    可以显著减少参数量并提升泛化能力。
    （Press & Wolf, 2016: "Using the Output Embedding to Improve Language Models"）
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 要求 embedding_size == hidden_size 才能 weight tying
        assert embedding_size == hidden_size, \
            f"Weight tying requires embedding_size ({embedding_size}) == hidden_size ({hidden_size})"

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output_dropout = nn.Dropout(dropout)

        # 使用 embedding 权重作为输出投影（weight tying）
        # 需要额外的 bias
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -init_scale, init_scale)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "lstm" in name and "bias_hh" in name:
                    n = param.size(0) // 4
                    param.data[n:2 * n].fill_(1.0)

    def forward(self, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape

        emb = self.embedding(input_ids)
        emb = self.embed_dropout(emb)

        lstm_out, hidden = self.lstm(emb, hidden)
        lstm_out = self.output_dropout(lstm_out)

        # Weight tying: 用 embedding 权重矩阵的转置做输出投影
        logits = F.linear(lstm_out, self.embedding.weight, self.output_bias)

        return logits, hidden

    def detach_hidden(self, hidden):
        if hidden is None:
            return None
        h, c = hidden
        return (h.detach(), c.detach())

    def init_hidden(self, batch_size, device="cpu"):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


# 默认模型类
Model = LSTMLanguageModel


if __name__ == "__main__":
    # 快速测试
    from config import config as cfg

    print(f"配置: vocab_size={cfg.vocab_size}, embedding={cfg.embedding_size}, "
          f"hidden={cfg.hidden_size}, layers={cfg.num_layers}")

    model = Model(
        vocab_size=cfg.vocab_size,
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        init_scale=cfg.init_scale,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试前向传播
    batch_size, seq_len = 20, 35
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, hidden = model(dummy_input)
    print(f"输入: {dummy_input.shape} → 输出: {logits.shape}")
    print(f"隐状态 h: {hidden[0].shape}, c: {hidden[1].shape}")
