# 训练监控 - TensorBoard

## 快速开始

### 1. 训练模型

```bash
python -m cs336_basics.train \
    --config configs/train_tinystories.json \
    --tensorboard
```

### 2. 查看TensorBoard

```bash
tensorboard --logdir runs/tinystories
```

访问 http://localhost:6006

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
