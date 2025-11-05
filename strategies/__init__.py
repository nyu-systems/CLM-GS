"""
Training Strategies for Gaussian Splatting
==========================================

This package contains different offloading strategies for training 3D Gaussian Splatting models:

- naive_offload: Simple baseline implementation with all parameters on CPU
- clm_offload: Cache-Locality-aware Memory offload with retention optimization
- (future) no_offload: Standard implementation with all parameters on GPU
"""

from . import naive_offload
from . import clm_offload

__all__ = ['naive_offload', 'clm_offload']

