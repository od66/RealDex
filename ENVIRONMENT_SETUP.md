# RealDex Authors Exact Environment Setup

This guide provides multiple methods to recreate the exact conda environment (`realdex_authors_exact`) used by the RealDex authors.

## Quick Setup (Recommended)

### Method 1: Automated Script

Run the automated setup script:

```bash
./recreate_realdex_authors_exact.sh
```

This script will:
- Create a new conda environment named `realdex_authors_exact`
- Install all dependencies with exact versions
- Install pytorch_kinematics from the local thirdparty directory
- Run comprehensive tests to verify the installation

### Method 2: Using Environment YAML

```bash
conda env create -f realdex_authors_exact_clean.yml
conda activate realdex_authors_exact
cd dexgrasp_generation/thirdparty/pytorch_kinematics
pip install -e .
cd ../../..
```

## Manual Setup

If you prefer to install step by step:

```bash
# Create environment
conda create -n realdex_authors_exact python=3.8.20 -y
conda activate realdex_authors_exact

# Install from frozen requirements
pip install -r requirements-frozen.txt

# Install pytorch_kinematics
cd dexgrasp_generation/thirdparty/pytorch_kinematics
pip install -e .
cd ../../..
```

## Key Package Versions

The environment includes these critical packages with exact versions:

- **Python**: 3.8.20
- **PyTorch**: 1.13.0+cu117 (with CUDA 11.7 support)
- **PyTorch3D**: 0.3.0 (exact authors' version)
- **PyTorch Geometric**: 2.2.0
- **NumPy**: 1.24.4
- **OpenCV**: 4.8.0.76
- **PyTorch Lightning**: 2.0.8

## Verification

After installation, verify everything works:

```bash
conda activate realdex_authors_exact
python -c "import torch, torch_geometric, pytorch3d, pytorch_kinematics; print('All core dependencies loaded successfully! 🚀')"
```

## GPU Testing

If you're on a cluster, test on a GPU node:

```bash
# Request GPU node (example for SLURM)
srun -p gpu --pty --nodes=1 --ntasks=1 --mem=8000 --time=24:00:00 --gres=gpu:1 /bin/bash

# Activate environment and test CUDA
conda activate realdex_authors_exact
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

1. **PyTorch3D Installation Fails**: 
   - Ensure you have the exact PyTorch version (1.13.0+cu117)
   - PyTorch3D 0.3.0 is specifically compatible with PyTorch 1.13.0

2. **CUDA Version Mismatch**:
   - The environment uses CUDA 11.7 compatible packages
   - Should work on most modern CUDA installations

3. **pytorch_kinematics Not Found**:
   - Make sure you're running the script from the RealDex root directory
   - The `dexgrasp_generation/thirdparty/pytorch_kinematics` directory must exist

### Package Sources

- PyTorch packages: `https://download.pytorch.org/whl/cu117`
- PyTorch Geometric extensions: `https://data.pyg.org/whl/torch-1.13.0+cu117.html`
- All other packages: Standard PyPI

## Files Created

This setup creates:
- `recreate_realdex_authors_exact.sh`: Automated setup script
- `realdex_authors_exact_clean.yml`: Clean conda environment file
- `ENVIRONMENT_SETUP.md`: This documentation

## Environment Comparison

| Environment | Purpose | PyTorch | PyTorch3D | Notes |
|-------------|---------|---------|-----------|-------|
| `realdex_authors_exact` | Exact reproduction | 1.13.0+cu117 | 0.3.0 | Frozen versions |
| `realdex` | Development | 1.13.0+cu117 | Latest | More flexible |

Choose `realdex_authors_exact` for exact reproducibility of the authors' results.
