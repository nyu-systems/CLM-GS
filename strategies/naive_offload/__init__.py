"""
Naive Offload Mode
==================

Simple "braindead" offload implementation for baseline comparison.
All parameters stored in CPU pinned memory, transferred to GPU for each batch.
"""

from .gaussian_model import GaussianModelNaiveOffload
from .engine import naive_offload_train_one_batch, naive_offload_eval_one_cam

__all__ = [
    'GaussianModelNaiveOffload',
    'naive_offload_train_one_batch',
    'naive_offload_eval_one_cam',
]
