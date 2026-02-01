# 训练监控 - TensorBoard

## 安装 TensorBoard

在使用 TensorBoard 之前，需要先安装：

```bash
pip install tensorboard
```

或使用 uv：

```bash
uv pip install tensorboard
```

## 快速开始

### 1. 训练模型

```bash
python -m cs336_basics.train \
    --config configs/train_tinystories.json \
    --tensorboard
```

**注意**：如果 TensorBoard 未安装，训练脚本会显示警告信息，但训练仍会正常进行（只是不会记录日志）。

### 2. 查看 TensorBoard

训练开始后，脚本会显示日志目录路径，例如：
```
TensorBoard logging enabled: runs/tinystories/tinystories-4L-512d
View with: tensorboard --logdir runs/tinystories
```

在另一个终端运行：

```bash
tensorboard --logdir runs/tinystories
```

然后在浏览器访问 http://localhost:6006

### 3. 快速演示

```bash
# 100步训练演示（1-2分钟）
uv run python cs336_basics/examples/quick_train_demo.py
```

## 配置文件

### configs/train_tinystories.json

```json
{
  "train_data": "data/tokenized/tinystories_train.uint16.npy",
  "val_data": "data/tokenized/tinystories_valid.uint16.npy",
  "vocab_size": 10000,
  "context_length": 256,
  "num_layers": 6,
  "d_model": 384,
  "num_heads": 6,
  "batch_size": 16,
  "max_iters": 10000,
  "tensorboard": true
}
```

## 记录的指标

- **train/loss** - 训练损失
- **train/loss_smoothed** - 平滑损失
- **train/lr** - 学习率
- **train/grad_norm** - 梯度范数
- **train/perplexity** - 困惑度
- **val/loss** - 验证损失
- **val/perplexity** - 验证困惑度

## 命令行参数

```bash
python -m cs336_basics.train \
    --train-data data/tokenized/tinystories_train.uint16.npy \
    --val-data data/tokenized/tinystories_valid.uint16.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --num-layers 6 \
    --d-model 384 \
    --batch-size 16 \
    --max-iters 10000 \
    --tensorboard \
    --run-name "my-experiment"
```

## 继续训练

```bash
python -m cs336_basics.train \
    --config configs/train_tinystories.json \
    --resume checkpoints/tinystories/checkpoint_5000.pt
```

## 测试

```bash
uv run python tests/test_training.py
```
