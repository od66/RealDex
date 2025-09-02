#!/bin/bash

# RealDex Environment Setup Script for Login Node
# Sets up conda environment with exact PyTorch3D compatibility
# Based on requirements-frozen.txt and compatibility research

set -e

echo "=== RealDex Environment Setup on Login Node ==="
echo "Current node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Available space: $(df -h /home/od66 | tail -1 | awk '{print $4}')"

# Environment name
ENV_NAME="realdex"

# Remove existing environment if it exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Removing existing $ENV_NAME environment..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating new conda environment: $ENV_NAME with Python 3.8..."
conda create -n $ENV_NAME python=3.8 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch 1.13.0 with CUDA 11.7 (CPU version for login node)..."
# Install PyTorch with CUDA 11.7 support (will work on both CPU and GPU)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo "Installing PyTorch3D 0.3.0 with proper compatibility..."
# Install PyTorch3D 0.3.0 - this version is compatible with PyTorch 1.13.0
pip install pytorch3d==0.3.0

echo "Installing PyTorch Geometric and related packages..."
pip install torch-geometric==2.2.0
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0+pt113cu117 --extra-index-url https://data.pyg.org/whl/torch-1.13.0+cu117.html

echo "Installing core scientific packages..."
pip install numpy==1.24.4 scipy==1.10.0 matplotlib==3.5.1 scikit-learn==1.3.0

echo "Installing computer vision and ML packages..."
pip install opencv-python==4.8.0.76
pip install pytorch-lightning==2.0.8 torchmetrics==1.5.2 lightning-utilities==0.11.9

echo "Installing configuration and utility packages..."
pip install hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3

echo "Installing 3D processing packages..."
pip install transforms3d==0.4.1 trimesh==3.9.8 transformations==2022.9.26

echo "Installing visualization and data packages..."
pip install plotly==5.14.1 healpy==1.16.2 astropy==5.2.2 h5py==3.11.0

echo "Installing remaining dependencies..."
pip install lxml==4.9.2 ipython==8.12.3 ipdb==0.13.13 tqdm==4.67.1 psutil==7.0.0
pip install absl-py==2.3.1 pyyaml==6.0.2 pillow==10.4.0 requests==2.32.4
pip install jinja2==3.1.6 packaging==25.0 typing-extensions==4.13.2
pip install fsspec==2025.3.0 aiohttp==3.10.11

echo "Installing fvcore and iopath (Meta packages for PyTorch3D)..."
pip install fvcore==0.1.5.post20221221 iopath==0.1.10 yacs==0.1.8

echo "Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "Testing PyTorch3D installation..."
python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}'); from pytorch3d.structures import Meshes; print('PyTorch3D structures import successful')"

echo "Testing other key imports..."
python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"
python -c "import trimesh; print(f'Trimesh version: {trimesh.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "=== Environment Setup Complete ==="
echo "Environment name: $ENV_NAME"
echo "To activate: conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "1. Test the environment on a GPU node"
echo "2. Run: srun -p gpu --pty --nodes=1 --ntasks=1 --mem=8000 --time=24:00:00 --gres=gpu:1 /bin/bash"
echo "3. Activate environment: conda activate $ENV_NAME"
echo "4. Test CUDA functionality"