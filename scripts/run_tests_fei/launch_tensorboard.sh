#!/bin/bash
# Launch TensorBoard to visualize training logs

# Find the most recent training log directory
LOG_DIR="logs/contrastive_train"

if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory $LOG_DIR not found!"
    echo "Please run training first."
    exit 1
fi

echo "Starting TensorBoard..."
echo "Log directory: $LOG_DIR"
echo ""
echo "Open your browser and navigate to: http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir="$LOG_DIR" --port=6006
