#!/bin/bash
# Run learning rate sweep in tmux with auto-shutdown after completion

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="lr_sweep"

echo "=========================================="
echo "Learning Rate Sweep - Tmux Session"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Session name: $SESSION_NAME"
echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: Tmux session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"

# Send command to tmux session to run lr_sweep.py directly
# The shutdown logic is now built into lr_sweep.py
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && python scripts/lr_sweep.py" C-m

echo ""
echo "Tmux session created and sweep started!"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (keep running):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To view logs:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "The system will automatically shutdown after training completes."
echo ""
