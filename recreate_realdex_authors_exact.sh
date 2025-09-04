#!/bin/bash

# Script to recreate the realdex_authors_exact conda environment
# This script recreates the exact environment used by the RealDex authors
# Based on the frozen environment.yml and requirements-frozen.txt

set -e

echo "=== RealDex Authors Exact Environment Recreation Script ==="
echo "Current node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Available space: $(df -h ~ | tail -1 | awk '{print $4}')"

# Environment name
ENV_NAME="realdex_authors_exact"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install conda/miniconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "🗑️  Removing existing $ENV_NAME environment..."
    conda env remove -n $ENV_NAME -y
fi

echo "🐍 Creating new conda environment: $ENV_NAME with Python 3.8.20..."
conda create -n $ENV_NAME python=3.8.20 -y

echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Verify we're in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "❌ Error: Failed to activate $ENV_NAME environment"
    exit 1
fi

echo "⚡ Installing PyTorch 1.13.0 with CUDA 11.7 support..."
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo "🔥 Installing PyTorch3D 0.3.0 (exact authors' version)..."
pip install pytorch3d==0.3.0

echo "🔗 Installing PyTorch Geometric ecosystem..."
pip install torch-geometric==2.2.0
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0+pt113cu117 --extra-index-url https://data.pyg.org/whl/torch-1.13.0+cu117.html

echo "🔬 Installing core scientific packages..."
pip install numpy==1.24.4 scipy==1.10.0 matplotlib==3.5.1 scikit-learn==1.3.0

echo "📷 Installing computer vision and ML packages..."
pip install opencv-python==4.8.0.76
pip install pytorch-lightning==2.0.8 torchmetrics==1.5.2 lightning-utilities==0.11.9

echo "⚙️  Installing configuration and utility packages..."
pip install hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3

echo "🎯 Installing 3D processing packages..."
pip install transforms3d==0.4.1 trimesh==3.9.8 transformations==2022.9.26

echo "📊 Installing visualization and data packages..."
pip install plotly==5.14.1 healpy==1.16.2 astropy==5.2.2 h5py==3.11.0

echo "🛠️  Installing development and utility packages..."
pip install lxml==4.9.2 ipython==8.12.3 ipdb==0.13.13 tqdm==4.67.1 psutil==7.0.0

echo "📦 Installing additional core dependencies..."
pip install absl-py==2.3.1 pyyaml==6.0.2 pillow==10.4.0 requests==2.32.4
pip install jinja2==3.1.6 packaging==25.0 typing-extensions==4.13.2
pip install fsspec==2025.3.0 aiohttp==3.10.11

echo "🏗️  Installing Meta packages for PyTorch3D compatibility..."
pip install fvcore==0.1.5.post20221221 iopath==0.1.10 yacs==0.1.8

echo "🤖 Installing pytorch_kinematics from local source..."
if [ -d "dexgrasp_generation/thirdparty/pytorch_kinematics" ]; then
    cd dexgrasp_generation/thirdparty/pytorch_kinematics
    pip install -e .
    cd ../../..
    echo "✅ pytorch_kinematics installed successfully"
else
    echo "⚠️  Warning: pytorch_kinematics directory not found. Skipping local installation."
    echo "   Please run this script from the RealDex root directory."
fi

echo ""
echo "🧪 Testing installation..."

echo "Testing PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"

echo "Testing PyTorch3D..."
python -c "
import pytorch3d
print(f'PyTorch3D version: {pytorch3d.__version__}')
from pytorch3d.structures import Meshes
print('PyTorch3D structures import: ✅')
"

echo "Testing PyTorch Geometric..."
python -c "
import torch_geometric
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
"

echo "Testing other key packages..."
python -c "
import trimesh
print(f'Trimesh version: {trimesh.__version__}')
import cv2
print(f'OpenCV version: {cv2.__version__}')
import numpy as np
print(f'NumPy version: {np.__version__}')
import scipy
print(f'SciPy version: {scipy.__version__}')
"

echo "Testing pytorch_kinematics (if available)..."
python -c "
try:
    import pytorch_kinematics
    print('pytorch_kinematics: ✅')
except ImportError as e:
    print(f'pytorch_kinematics: ⚠️  Not available ({e})')
"

echo ""
echo "🎉 Environment Recreation Complete!"
echo "Environment name: $ENV_NAME"
echo "To activate: conda activate $ENV_NAME"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. If on a cluster, test on a GPU node:"
echo "   srun -p gpu --pty --nodes=1 --ntasks=1 --mem=8000 --time=24:00:00 --gres=gpu:1 /bin/bash"
echo "3. Test CUDA functionality with your RealDex code"
echo ""
echo "🔍 To verify all imports work:"
echo "python -c \"import torch, torch_geometric, pytorch3d, pytorch_kinematics; print('All core dependencies loaded successfully! 🚀')\""
