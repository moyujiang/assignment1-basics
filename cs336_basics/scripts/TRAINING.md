# Training Script

简洁优雅的 Transformer 语言模型训练脚本，支持：
- ✅ 完整超参数配置
- ✅ 内存高效的 memmap 数据加载
- ✅ 自动检查点保存/恢复
- ✅ 训练和验证性能日志

## 快速开始

### 使用配置文件训练

```bash
# TinyStories 小模型（6层，384维）
uv run python -m cs336_basics.train --config configs/tinystories_small.json

# GPT-2 124M 规模模型（12层，768维）
uv run python -m cs336_basics.train --config configs/gpt2_124M.json
```

### 命令行参数训练

```bash
uv run python -m cs336_basics.train \
  --train-data data/tokenized/tinystories_train.uint16.npy \
  --val-data data/tokenized/tinystories_valid.uint16.npy \
  --vocab-size 10000 \
  --num-layers 6 \
  --d-model 384 \
  --num-heads 6 \
  --batch-size 16 \
  --max-iters 10000
```

### 从检查点恢复训练

```bash
uv run python -m cs336_basics.train \
  --config configs/tinystories_small.json \
  --resume checkpoints/tinystories_small/checkpoint_2000.pt
```

## 配置参数

### 数据
- `--train-data`: 训练数据路径 (.npy 格式，memmap 自动启用)
- `--val-data`: 验证数据路径
- `--vocab-size`: 词表大小

### 模型架构
- `--context-length`: 最大序列长度 (默认: 1024)
- `--num-layers`: Transformer 层数 (默认: 12)
- `--d-model`: 模型维度 (默认: 768)
- `--num-heads`: 注意力头数 (默认: 12)
- `--d-ff`: FFN 隐藏层维度 (默认: 3072)
- `--rope-theta`: RoPE theta 参数 (默认: 10000.0)

### 优化器
- `--max-lr`: 最大学习率 (默认: 3e-4)
- `--min-lr`: 最小学习率 (默认: 3e-5)
- `--weight-decay`: 权重衰减 (默认: 0.1)
- `--beta1`, `--beta2`: AdamW beta 参数
- `--grad-clip`: 梯度裁剪 (默认: 1.0，设为 0 禁用)

### 训练
- `--batch-size`: 批大小 (默认: 8)
- `--max-iters`: 最大迭代次数 (默认: 100000)
- `--warmup-iters`: 学习率预热步数 (默认: 2000)

### 日志和检查点
- `--log-interval`: 日志输出间隔 (默认: 10)
- `--eval-interval`: 验证间隔 (默认: 500)
- `--checkpoint-dir`: 检查点目录 (默认: checkpoints)
- `--checkpoint-interval`: 检查点保存间隔 (默认: 5000)

## 输出示例

```
================================================================================
Training Configuration
================================================================================
batch_size               : 16
checkpoint_dir           : checkpoints/tinystories_small
context_length           : 256
d_ff                     : 1536
d_model                  : 384
...
================================================================================
Loading datasets...
Train size: 45,123,456 tokens | Val size: 2,234,567 tokens

Initializing model...
Model parameters: 21,234,816

Starting training...
iter      0 | loss 10.2341 | lr 0.00e+00 | 123.45ms
iter     10 | loss  9.8234 | lr 3.00e-06 | 98.76ms
...
[EVAL] iter    500 | val_loss 8.1234 | val_ppl 3345.67
[CHECKPOINT] Saved to checkpoints/tinystories_small/checkpoint_2000.pt
...
```

## 内存效率

脚本使用 `np.memmap` 自动加载数据，无需将整个数据集载入内存：

```python
# data.py 自动处理
train_data = load_dataset(path, vocab_size, use_mmap=True)  # 仅映射，不加载
inputs, targets = get_batch(train_data, batch_size, context_length, device)  # 按需读取
```

## 检查点格式

检查点包含完整的训练状态：

```python
{
    'model_state_dict': {...},      # 模型参数
    'optimizer_state_dict': {...},  # 优化器状态（包括动量）
    'iteration': 5000               # 当前迭代次数
}
```

## 扩展：Weights & Biases 集成

### 安装 wandb

```bash
uv pip install wandb
wandb login
```

### 启用 wandb 日志

**方法 1：命令行参数**

```bash
uv run python -m cs336_basics.train \
  --config configs/tinystories_small.json \
  --wandb \
  --wandb-project "my-transformers" \
  --wandb-run-name "exp-1-tinystories" \
  --wandb-tags "baseline,tinystories"
```

**方法 2：配置文件**

编辑 `configs/tinystories_small.json`：

```json
{
  ...
  "wandb": true,
  "wandb_project": "cs336-transformer",
  "wandb_run_name": "tinystories-small-6L-384d",
  "wandb_tags": "baseline,small"
}
```

然后运行：

```bash
uv run python -m cs336_basics.train --config configs/tinystories_small.json
```

### 记录的指标

训练脚本自动记录以下指标到 W&B：

**训练指标**（每 `log_interval` 步）：
- `train/loss` - 交叉熵损失
- `train/lr` - 当前学习率
- `train/iter_time_ms` - 每步耗时（毫秒）

**验证指标**（每 `eval_interval` 步）：
- `val/loss` - 验证集交叉熵损失
- `val/perplexity` - 验证集困惑度

所有超参数也会自动记录到 W&B config。

### W&B 参数

- `--wandb` - 启用 W&B 日志（默认: False）
- `--wandb-project` - W&B 项目名称（默认: "cs336-transformer"）
- `--wandb-run-name` - 运行名称（默认: None，自动生成）
- `--wandb-tags` - 逗号分隔的标签（默认: None）
