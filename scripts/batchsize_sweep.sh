#!/bin/bash
#SBATCH -J ts_batchsize_sweep
#SBATCH -p lfs-dev-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o scripts/logs/%A_%a.out
#SBATCH -e scripts/logs/%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# When running under Slurm, the script may be copied to a spool directory.
# Prefer SLURM_SUBMIT_DIR (directory where `sbatch` was invoked).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	REPO_ROOT="$SLURM_SUBMIT_DIR"
else
	REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

mkdir -p scripts/logs

echo "Starting TinyStories batch-size sweep (bash)"

# Disable output buffering for SLURM logs
export PYTHONUNBUFFERED=1

# Helps reduce CUDA memory fragmentation for some workloads.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if command -v nvidia-smi >/dev/null 2>&1; then
	echo "GPU status (nvidia-smi):"
	nvidia-smi || true
	printf '%80s\n' '' | tr ' ' '-'
fi

BATCH_SIZES=(16 32 64 128)

# If running as a Slurm job array, each task runs exactly one batch size.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
	idx="$SLURM_ARRAY_TASK_ID"
	if (( idx < 0 || idx >= ${#BATCH_SIZES[@]} )); then
		echo "ERROR: SLURM_ARRAY_TASK_ID=$idx out of range" >&2
		exit 2
	fi

	bs="${BATCH_SIZES[$idx]}"
	results_file="scripts/logs/batchsize_sweep_results_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
	echo "Array task ${SLURM_ARRAY_TASK_ID}: batch_size=${bs}"

	uv run python -u scripts/batchsize_sweep.py \
		--config configs/train_tinystories.json \
		--batch-sizes "$bs" \
		--total-train-tokens 327680000 \
		--results-file "$results_file" \
		"$@"
	
	echo "Done: batch_size=${bs} (results: ${results_file})"
	exit 0
fi

# Non-array mode (e.g., running locally): run all batch sizes sequentially.
results_file="scripts/logs/batchsize_sweep_results_${SLURM_JOB_ID:-local}.json"
uv run python -u scripts/batchsize_sweep.py \
	--config configs/train_tinystories.json \
	--batch-sizes "${BATCH_SIZES[@]}" \
	--total-train-tokens 327680000 \
	--results-file "$results_file" \
	"$@"

echo "Sweep completed (results: ${results_file})"
