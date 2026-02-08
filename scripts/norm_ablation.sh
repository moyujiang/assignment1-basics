#!/bin/bash
#SBATCH -J norm_ablation
#SBATCH -p lfs-dev-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --chdir=/home/stu2400012941/assignment1-basics
#SBATCH -o scripts/logs/%j.out
#SBATCH -e scripts/logs/%j.err

set -euo pipefail

# Ensure logs directory exists
mkdir -p scripts/logs

echo "Starting training"

# Use PYTHONUNBUFFERED=1 to disable output buffering for SLURM logs
# uv run will use the shebang in the script, so we can run it directly
PYTHONUNBUFFERED=1 uv run --no-sync python -m cs336_basics.train \
	--config configs/train_tinystories.json \
	--no-rmsnorm \
	--max-lr 5e-4 \
	--tag no-rmsnorm

echo "Training completed"