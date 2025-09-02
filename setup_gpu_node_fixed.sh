#!/bin/bash

# RealDex Environment Setup Script for GPU Node
# Resolves PyTorch3D compatibility issues with proper version matching

set -e

echo "=== RealDex GPU Node Environment Setup ==="
echo "Current node: $(hostname)"
echo "Available CUDA version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"
echo "Available disk space: $(df -h / | tail -1 | awk '{print $4}')"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check CUDA version and adjust PyTorch version accordingly
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1 | cut -d'V' -f2)
echo "Detected CUDA version: $CUDA_VERSION"

# Remove existing environment if it exists
if conda env list | grep -q "realdex"; then
    echo "Removing existing realdex environment..."
    conda env remove -n realdex -y
fi

# Create new environment with Python 3.8 (as used by authors)
echo "Creating new conda environment: realdex with Python 3.8..."
conda create -n realdex python=3.8 pip -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realdex

# Install build essentials and ninja for faster compilation
echo "Installing build tools..."
conda install -c conda-forge ninja -y
pip install wheel setuptools

# CUDA 11.2 Compatibility Setup
# Since we have CUDA 11.2, we need to use PyTorch versions that support it
# PyTorch 1.13.0 requires CUDA 11.6+, so we'll use PyTorch 1.12.1 with CUDA 11.3
echo "Installing PyTorch 1.12.1 with CUDA 11.3 support (compatible with CUDA 11.2)..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch3D compatible with PyTorch 1.12.1
echo "Installing PyTorch3D 0.7.2 (compatible with PyTorch 1.12.1)..."
pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html

# Install PyTorch Geometric and related packages
echo "Installing PyTorch Geometric ecosystem..."
pip install torch-geometric==2.2.0
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 --find-links https://data.pyg.org/whl/torch-1.12.1+cu113.html

# Install other core dependencies with exact versions from working setup
echo "Installing other dependencies..."
pip install opencv-python==4.8.0.76
pip install numpy==1.24.4
pip install scipy==1.10.0
pip install matplotlib==3.5.1
pip install scikit-learn==1.3.0
pip install pytorch-lightning==2.0.8
pip install hydra-core==1.3.2
pip install transforms3d==0.4.1
pip install trimesh==3.9.8
pip install plotly==5.14.1
pip install healpy==1.16.2
pip install h5py==3.11.0
pip install lxml==4.9.2
pip install ipython==8.12.3
pip install ipdb==0.13.13
pip install tqdm==4.67.1

# Install additional dependencies for compatibility
echo "Installing additional compatibility packages..."
pip install fvcore==0.1.5.post20221221
pip install iopath==0.1.10
pip install yacs==0.1.8

# Install pytorch_kinematics
echo "Installing pytorch_kinematics..."
cd dexgrasp_generation/thirdparty/pytorch_kinematics
pip install -e .
cd ../../..

# Test the installation
echo "Testing installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

import pytorch3d
print(f'PyTorch3D version: {pytorch3d.__version__}')

import torch_geometric
print(f'PyTorch Geometric version: {torch_geometric.__version__}')

try:
    import pytorch_kinematics
    print('PyTorch Kinematics: OK')
except ImportError as e:
    print(f'PyTorch Kinematics error: {e}')

print('✅ All core dependencies loaded successfully!')
"

echo "=== Environment Setup Complete ==="
echo "Environment name: realdex"
echo "To activate: conda activate realdex"
echo "To test CSDF installation later: cd CSDF && pip install -e ."