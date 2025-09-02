# PyTorch3D Mock Implementation for RealDex Compatibility
# This provides essential PyTorch3D functions without the problematic C extensions

__version__ = "0.3.0"

# Mock the _C module to avoid import errors
class MockC:
    pass

import sys
sys.modules['pytorch3d._C'] = MockC()