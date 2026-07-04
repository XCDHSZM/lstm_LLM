# LSTM 神经网络语言模型 —— PyTorch 实现

基于 Zaremba et al. "Recurrent Neural Network Regularization" (2015) 的 LSTM 语言模型，使用 PTB (Penn Treebank) 数据集进行训练和评估。

## 项目结构

```
lstm_LLM/
├── run.py                  # 一键运行入口
├── download_data.py        # PTB 数据下载脚本
├── requirements.txt        # 依赖 (torch, numpy, tqdm)
├── data/
│   ├── raw/               # 原始 PTB 语料 (需要下载)
│   └── processed/         # 预处理后的 .npy / .json
├── models/                # 训练好的模型权重
└── src/
    ├── config.py          # 超参数 (Small/Medium/Large 三套配置)
    ├── data_loader.py     # 数据预处理 + BPTT batch 生成
    ├── model.py           # LSTM 语言模型 (支持 weight tying)
    ├── train.py           # BPTT 训练循环 (梯度截断 + 学习率衰减)
    └── evaluate.py        # 困惑度 (PPL) 评估
```

## 运行方式

在当前目录下，使用 **cmd** 或 **PowerShell** 逐步执行：

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 PTB 数据

```bash
python download_data.py
```

如果自动下载失败，手动下载 `simple-examples.tgz` 解压到 `data/raw/`：

- http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

解压后 `data/raw/` 应包含以下文件：

- `ptb.train.txt`
- `ptb.valid.txt`
- `ptb.test.txt`

### 3. 训练模型

```bash
python run.py train
```

### 4. 评估模型

```bash
python run.py evaluate
```

### 一键完整流程

```bash
python run.py all      # 数据预处理 → 训练 → 评估
```

### 使用 CPU 训练（如果没有 GPU）

```bash
python run.py train --cpu
```

## 关键设计说明

| 模块 | 说明 |
|------|------|
| **config.py** | 提供 Small/Medium/Large 三套配置（参考 Zaremba 2015），当前默认 **MediumConfig** |
| **data_loader.py** | BPTT 标准 batch 生成：数据 reshape 为 `[batch_size, -1]`，按 `num_steps` 滑动窗口 |
| **model.py** | 两种实现：`LSTMLanguageModel`（标准）和 `LSTMLanguageModelTied`（weight tying 减少参数） |
| **train.py** | SGD 优化器 + StepLR 衰减 + 梯度截断 `max_grad_norm=5.0` + 隐状态 detach |
| **evaluate.py** | 加载 best_model.pt 计算验证集/测试集 PPL |

## 配置切换

编辑 `src/config.py`，修改最后的配置选择：

```python
# config = SmallConfig()
config = MediumConfig()
# config = LargeConfig()
```

| 配置 | hidden_size | num_layers | batch_size | num_steps | 参数量 |
|------|-------------|------------|------------|-----------|--------|
| Small | 200 | 2 | 20 | 20 | ~6M |
| Medium | 650 | 2 | 20 | 35 | ~20M |
| Large | 1500 | 2 | 20 | 35 | ~66M |

## 预期效果

| 配置 | 验证 PPL | 测试 PPL |
|------|----------|----------|
| Small | ~110-130 | ~105-120 |
| Medium | ~80-100 | ~78-95 |
| Large | ~75-85 | ~72-78 |

> 注意：实际效果受训练 epoch 数、随机种子等因素影响。Large 配置需要 GPU 显存 ≥ 8GB。

## 依赖

- Python >= 3.8
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- tqdm >= 4.60.0
