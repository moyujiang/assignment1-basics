#!/bin/bash
#SBATCH -J owt_lr_sweep
#SBATCH -p lfs-dev-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o scripts/logs/%j.out
#SBATCH -e scripts/logs/%j.err

set -euo pipefail

# When running under Slurm, the script may be copied to a spool directory.
# In that case, ${BASH_SOURCE[0]} points to the copied script, not the repo.
# Prefer SLURM_SUBMIT_DIR (directory where `sbatch` was invoked).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	REPO_ROOT="$SLURM_SUBMIT_DIR"
else
	REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$REPO_ROOT"

mkdir -p scripts/logs

echo "Starting OpenWebText LR sweep (bash)"

if command -v nvidia-smi >/dev/null 2>&1; then
	echo "GPU status (nvidia-smi):"
	nvidia-smi || true
	printf '%80s\n' '' | tr ' ' '-'
fi

# Disable output buffering for SLURM logs
export PYTHONUNBUFFERED=1

# Helps reduce CUDA memory fragmentation for some workloads.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BASE_CONFIG="configs/train_openwebtext.json"

# Default learning rates (override by passing positional args).
LRS_DEFAULT=(8e-3)

EXTRA_ARGS=()

LRS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--config)
			BASE_CONFIG="$2"; shift 2 ;;
		--)
			shift
			EXTRA_ARGS=("$@");
			break
			;;
		-h|--help)
			cat <<'EOF'
Usage:
	bash scripts/run_owt_lr.sh [lr1 lr2 ...] [-- --extra train args]

Options:
	--config PATH         Base JSON config (default: configs/train_openwebtext.json)
	--                   Everything after is passed to cs336_basics.train

Examples:
	sbatch scripts/run_owt_lr.sh
	bash scripts/run_owt_lr.sh 3e-4 6e-4 1e-3
	bash scripts/run_owt_lr.sh -- --device cpu
EOF
			exit 0
			;;
		*)
			LRS+=("$1")
			shift
			;;
	esac
done

if [[ ${#LRS[@]} -eq 0 ]]; then
	LRS=("${LRS_DEFAULT[@]}")
fi

echo "Base config: $BASE_CONFIG"
echo "Learning rates: ${LRS[*]}"

for lr in "${LRS[@]}"; do
	printf '%80s\n' '' | tr ' ' '='
	echo "Running: max_lr=$lr"
	printf '%80s\n' '' | tr ' ' '='

	CMD=(uv run python -u -m cs336_basics.train --config "$BASE_CONFIG" --max-lr "$lr" --tensorboard)

	if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
		CMD+=("${EXTRA_ARGS[@]}")
	fi

	echo "Command: ${CMD[*]}"
	"${CMD[@]}"
done

echo "OWT LR sweep completed"
