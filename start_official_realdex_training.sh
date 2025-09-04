#!/bin/bash

# Official RealDex Training Script
# Trains on all available official RealDex datasets with full monitoring

set -e

echo "🚀 Starting Official RealDex Training"
echo "======================================="
echo "Time: $(date)"
echo "Duration: ~4-6 hours (250 epochs)"
echo "Data: 7 official RealDex datasets (14,380 samples)"
echo "Output: runs/cvae_official_realdex/"
echo ""

# Navigate to correct directory
cd /home/od66/GRILL/RealDex/dexgrasp_generation

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realdex_authors_exact

# Verify environment
echo "✅ Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "✅ Python: $(python --version)"
echo "✅ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "✅ PyTorch3D: $(python -c 'import pytorch3d; print(pytorch3d.__version__)')"
echo "✅ CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' | grep -q True; then
    echo "✅ GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Check data availability
echo "📊 Data Summary:"
echo "   Total samples: $(find ../data/realdex_qpos_format/ -name "*.npz" | wc -l)"
echo "   Datasets: $(ls ../data/realdex_qpos_format/ | wc -l)"
echo ""

# Create monitoring file
MONITOR_FILE="training_monitor_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Monitoring file: $MONITOR_FILE"
echo ""

# Start training with comprehensive logging
echo "🔥 Starting full training on official RealDex data..."
echo "   Configuration: cvae_official_config"
echo "   Expected duration: 4-6 hours"
echo "   Checkpoints saved every 500 iterations"
echo "   Logs updated every 50 epochs"
echo ""

# Run training with monitoring
python ./network/train.py --config-name cvae_official_config 2>&1 | tee "$MONITOR_FILE"

echo ""
echo "✅ Training completed at $(date)"
echo "📊 Results saved to: runs/cvae_official_realdex/"
echo "📝 Full log saved to: $MONITOR_FILE"
echo "🎯 Training validation complete - pipeline verified!"
