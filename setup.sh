#!/bin/bash

set -e

if ! command -v conda &> /dev/null; then
    echo "conda not found"
    exit 1
fi

if conda env list | grep -q "realdex"; then
    conda env remove -n realdex -y
fi
conda create -n realdex python=3.8 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realdex

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1130/download.html
pip install torch-geometric==2.2.0
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install opencv-python==4.8.0.76 hydra-core==1.3.2 transforms3d==0.4.1 trimesh==3.9.8 scipy==1.10.0 matplotlib==3.5.1 scikit-learn==1.3.0 pytorch-lightning==2.0.8
pip install lxml==4.9.2 healpy==1.16.2 plotly==5.14.1 ipython ipdb h5py tqdm

cd dexgrasp_generation/thirdparty/pytorch_kinematics
pip install -e .
cd ../../..

python -c "import torch, torch_geometric, pytorch3d, pytorch_kinematics; print('done')"
