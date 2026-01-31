#!/bin/bash
# 使用 Weights & Biases 记录训练过程的示例脚本

# 确保安装了 wandb
echo "检查 wandb 安装..."
uv pip install wandb -q

# 登录 wandb（如果还未登录）
# wandb login

# 方法 1：使用配置文件 + 命令行覆盖
echo "启动训练（方法 1：配置文件 + 命令行参数）..."
uv run python -m cs336_basics.train \
  --config configs/tinystories_small.json \
  --max-iters 1000 \
  --wandb \
  --wandb-project "cs336-experiments" \
  --wandb-run-name "tinystories-baseline" \
  --wandb-tags "baseline,tinystories,6L"

# 方法 2：纯命令行参数
echo "启动训练（方法 2：纯命令行）..."
uv run python -m cs336_basics.train \
  --train-data data/tokenized/tinystories_train.uint16.npy \
  --val-data data/tokenized/tinystories_valid.uint16.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --num-layers 6 \
  --d-model 384 \
  --num-heads 6 \
  --d-ff 1536 \
  --batch-size 16 \
  --max-iters 1000 \
  --wandb \
  --wandb-project "cs336-experiments" \
  --wandb-run-name "tinystories-custom"
