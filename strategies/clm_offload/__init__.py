"""
CLM Offload Mode
================

Cache-Locality-aware Memory offload implementation.
Uses retention-based parameter offloading with TSP-based camera ordering
to maximize parameter reuse across consecutive frames.
"""

from .gaussian_model import GaussianModelCLMOffload
from .engine import clm_offload_train_one_batch, clm_offload_eval_one_cam

__all__ = [
    'GaussianModelCLMOffload',
    'clm_offload_train_one_batch',
    'clm_offload_eval_one_cam',
]
