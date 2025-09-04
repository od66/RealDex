#!/bin/bash

# Training Monitor Script
# Monitors the progress of RealDex training

TRAINING_DIR="/home/od66/GRILL/RealDex/dexgrasp_generation/runs/cvae_official_realdex"
LOG_FILE="$TRAINING_DIR/log.txt"

echo "🔍 RealDex Training Monitor"
echo "=========================="
echo "Time: $(date)"
echo ""

if [ ! -d "$TRAINING_DIR" ]; then
    echo "❌ Training directory not found: $TRAINING_DIR"
    echo "   Training may not have started yet."
    exit 1
fi

echo "📁 Training Directory: $TRAINING_DIR"
echo "📊 Directory Size: $(du -sh $TRAINING_DIR | cut -f1)"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "📝 Latest Log Entries:"
    echo "---------------------"
    tail -20 "$LOG_FILE"
    echo ""
    
    # Extract training progress
    EPOCHS=$(grep -o "epoch [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
    echo "🎯 Current Progress: Epoch $EPOCHS/250"
    
    # Calculate progress percentage
    if [ "$EPOCHS" -gt 0 ]; then
        PROGRESS=$((EPOCHS * 100 / 250))
        echo "📈 Progress: $PROGRESS%"
    fi
else
    echo "⚠️  Log file not found: $LOG_FILE"
    echo "   Training may still be initializing."
fi

echo ""
echo "📂 Checkpoints:"
if [ -d "$TRAINING_DIR/ckpt" ]; then
    ls -la "$TRAINING_DIR/ckpt/" | tail -5
    CHECKPOINT_COUNT=$(ls "$TRAINING_DIR/ckpt/"*.pt 2>/dev/null | wc -l || echo "0")
    echo "   Total checkpoints: $CHECKPOINT_COUNT"
else
    echo "   No checkpoints directory found yet."
fi

echo ""
echo "🔄 To check again: ./monitor_training.sh"
echo "🖥️  To attach to tmux: tmux attach -t official_realdex_training"
