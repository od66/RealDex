#!/bin/bash

# RealDex Validation Training Script
# Runs a 1-hour test training to validate the pipeline

set -e

echo "🚀 Starting RealDex Validation Training"
echo "Time: $(date)"
echo "Duration: ~1 hour (50 epochs instead of 250)"
echo "Data: driller_0 and driller_8 datasets"
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
echo ""

# Start training with logging
echo "🔥 Starting training..."
python ./network/train.py --config-name cvae_test_config 2>&1 | tee training_validation.log

echo ""
echo "✅ Training completed at $(date)"
echo "📊 Check results in: runs/cvae_test_1hr/"
echo "📝 Full log saved to: training_validation.log"
