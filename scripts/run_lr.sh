#!/bin/bash
#SBATCH -J lr_sweep
#SBATCH -p lfs-dev-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o scripts/logs/%j.out
#SBATCH -e scripts/logs/%j.err

cd ~/assignment1-basics

# Ensure logs directory exists
mkdir -p scripts/logs

echo "Starting training"

# Use PYTHONUNBUFFERED=1 to disable output buffering for SLURM logs
# uv run will use the shebang in the script, so we can run it directly
PYTHONUNBUFFERED=1 uv run python -u scripts/lr_sweep.py

echo "Training completed"