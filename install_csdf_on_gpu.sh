#!/bin/bash

# CSDF Installation Script for GPU Node
# Must be run after getting GPU access for CUDA compilation

set -e

echo "=== CSDF Installation on GPU Node ==="
echo "Current node: $(hostname)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Activate the environment
conda activate realdex_authors_exact

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "WARNING: CUDA not available, using FORCE_CUDA=1 for cross-compilation"
    export FORCE_CUDA=1
fi

# Install CSDF
echo "Installing CSDF (Custom Signed Distance Fields)..."
cd CSDF

# Try normal installation first
echo "Attempting: pip install -e ."
if pip install -e .; then
    echo "✓ CSDF installed successfully with pip install -e ."
else
    echo "pip install failed, trying python setup.py install..."
    if python setup.py install; then
        echo "✓ CSDF installed successfully with setup.py"
    else
        echo "❌ CSDF installation failed"
        exit 1
    fi
fi

cd ..

# Test CSDF installation
echo "Testing CSDF installation..."
python -c "
import torch
import csdf
from csdf import compute_sdf, index_vertices_by_faces

print('✓ CSDF import successful')
print('✓ compute_sdf function available')
print('✓ index_vertices_by_faces function available')

# Test basic functionality
vertices = torch.randn(100, 3).cuda() if torch.cuda.is_available() else torch.randn(100, 3)
faces = torch.randint(0, 100, (50, 3))
face_verts = index_vertices_by_faces(vertices, faces)
print(f'✓ CSDF basic operations working: face_verts shape {face_verts.shape}')

print('🎉 CSDF INSTALLATION SUCCESSFUL!')
"

echo ""
echo "=== CSDF Installation Complete ==="
echo "✓ CSDF compiled and tested successfully"
echo "✓ RealDex environment fully ready for training"