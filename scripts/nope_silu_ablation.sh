#!/bin/bash
#SBATCH -J nope_silu_lr
#SBATCH -p lfs-dev-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --chdir=/home/stu2400012941/assignment1-basics
#SBATCH -o /home/stu2400012941/assignment1-basics/scripts/logs/%x-%A_%a.out
#SBATCH -e /home/stu2400012941/assignment1-basics/scripts/logs/%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# NOTE: Using absolute paths for --output/--error so SLURM can open files even if
# the submit directory differs. Ensure this directory exists before sbatch.
mkdir -p /home/stu2400012941/assignment1-basics/scripts/logs

echo "Job: ${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array_id=${SLURM_ARRAY_TASK_ID}"

CONFIG="${CONFIG:-configs/train_tinystories.json}"

# Two learning rates you requested.
LR_A="${LR_A:-5e-3}"
LR_B="${LR_B:-1e-3}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LR="${LR_A}"
    COND="nope"
    EXTRA_ARGS="--no-rope"
    ;;
  1)
    LR="${LR_B}"
    COND="nope"
    EXTRA_ARGS="--no-rope"
    ;;
  2)
    LR="${LR_A}"
    COND="silu"
    EXTRA_ARGS="--ffn-type silu"
    ;;
  3)
    LR="${LR_B}"
    COND="silu"
    EXTRA_ARGS="--ffn-type silu"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
 esac

TAG="${COND}_lr${LR}"
# Make tag path-safe (train.py sanitizes too, but do it here for readability)
TAG="${TAG//\//-}"
TAG="${TAG//\\/-}"
TAG="${TAG// /-}"

echo "Condition: ${COND}  LR=${LR}  CONFIG=${CONFIG}  TAG=${TAG}"

PYTHONUNBUFFERED=1 uv run --no-sync python -m cs336_basics.train \
  --config "${CONFIG}" \
  --max-lr "${LR}" \
  --tag "${TAG}" \
  ${EXTRA_ARGS}
