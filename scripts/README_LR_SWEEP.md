# 学习率超参数搜索使用指南

## 概述

本脚本系统用于对 TinyStories 训练进行学习率超参数搜索，自动运行多个不同学习率的训练，并记录结果供后续分析。

## 学习率设计

基于已完成的基准训练（max_lr = 6e-4），我们设计了以下学习率进行测试：

- **3e-4** (0.5x 基准) - 较低学习率，更稳定但可能收敛慢
- **4e-4** (0.67x 基准) - 中等偏低学习率
- **6e-4** (1.0x 基准) - 基准学习率（已完成）
- **8e-4** (1.33x 基准) - 中等偏高学习率
- **1e-3** (1.67x 基准) - 较高学习率
- **1.2e-3** (2.0x 基准) - 高学习率，可能不稳定

所有学习率使用 `min_lr = max_lr / 10` 的默认设置。

## 使用步骤

### 1. 重命名现有目录（首次运行）

如果已经有使用基准学习率（6e-4）训练的结果，需要先重命名目录以符合新的命名规范：

```bash
python scripts/rename_existing_dirs.py
```

这将把：
- `checkpoints/tinystories` → `checkpoints/tinystories_lr6e-4`
- `runs/tinystories/tinystories-4L-512d` → `runs/tinystories/tinystories-4L-512d_lr6e-4`

### 2. 运行学习率搜索

#### 方式 1: 在 tmux 中运行（推荐，支持自动关机）

```bash
bash scripts/run_lr_sweep_tmux.sh
```

这将：
- 创建一个名为 `lr_sweep` 的 tmux 会话
- 自动运行所有学习率的训练
- 训练完成后自动关机

**查看运行状态：**
```bash
tmux attach -t lr_sweep
```

** detached（保持后台运行）：**
按 `Ctrl+B`，然后按 `D`

#### 方式 2: 直接运行（用于测试）

```bash
bash scripts/run_lr_sweep.sh
```

或直接运行 Python 脚本：

```bash
python scripts/lr_sweep.py
```

### 3. 查看结果

训练过程中，结果会实时保存到 `lr_sweep_results.json` 文件中。

训练完成后，每个学习率的模型和日志会保存在：

- **Checkpoints**: `checkpoints/tinystories_lr{lr}/`
- **TensorBoard logs**: `runs/tinystories/tinystories-4L-512d_lr{lr}/`

### 4. 查看 TensorBoard

查看所有学习率的训练曲线：

```bash
tensorboard --logdir runs/tinystories
```

然后在浏览器访问 http://localhost:6006

## 结果文件格式

`lr_sweep_results.json` 包含每个学习率的训练结果：

```json
[
  {
    "lr": 3e-4,
    "lr_str": "3e-4",
    "min_lr": 3e-5,
    "status": "completed",
    "elapsed_time": 12345.67,
    "checkpoint_dir": "checkpoints/tinystories_lr3e-4",
    "final_checkpoint": "checkpoints/tinystories_lr3e-4/checkpoint_final.pt",
    "tensorboard_dir": "runs/tinystories/tinystories-4L-512d_lr3e-4"
  },
  ...
]
```

## 注意事项

1. **训练时间**: 每个学习率需要训练 40,000 步，预计每个需要数小时（取决于硬件）
2. **磁盘空间**: 确保有足够的磁盘空间存储所有 checkpoints 和日志
3. **跳过已完成的训练**: 如果某个学习率已经训练完成，可以手动编辑 `scripts/lr_sweep.py` 中的 `LEARNING_RATES` 列表，移除已完成的学习率
4. **自动关机**: tmux 脚本会在所有训练完成后自动关机，请确保已保存所有重要数据

## 修改学习率列表

如果需要修改要测试的学习率，编辑 `scripts/lr_sweep.py` 文件中的 `LEARNING_RATES` 变量：

```python
LEARNING_RATES = [3e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1.2e-3]
```

## 故障排除

### 训练失败

如果某个学习率的训练失败，脚本会继续运行下一个学习率。检查失败原因：

1. 查看 tmux 会话的输出
2. 检查磁盘空间是否充足
3. 检查 GPU 内存是否足够

### 恢复训练

如果训练中断，可以手动恢复某个学习率的训练：

```bash
python -m cs336_basics.train \
    --config configs/train_tinystories.json \
    --max-lr 8e-4 \
    --min-lr 8e-5 \
    --tensorboard \
    --resume checkpoints/tinystories_lr8e-4/checkpoint_20000.pt
```

## 后续分析

训练完成后，可以使用以下方式分析结果：

1. **TensorBoard**: 查看所有学习率的训练曲线对比
2. **提取最终验证损失**: 从 `lr_sweep_results.json` 或 TensorBoard 日志中提取
3. **选择最佳模型**: 根据验证损失选择最佳学习率对应的模型
