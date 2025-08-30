# Setup

## Quick
```bash
./setup.sh
```

## Manual
```bash
conda create -n realdex python=3.8 pip -y
conda activate realdex
pip install -r requirements-frozen.txt
cd dexgrasp_generation/thirdparty/pytorch_kinematics
pip install -e .
```

## Test
```bash
python -c "import torch, torch_geometric, pytorch3d, pytorch_kinematics; print('works')"
```
