# Overview of `--no_offload`

Please read the **Common Setup** section in the main README.md first to understand the common implementations in all three stretegies. 

This strategy trains 3D Gaussian Splatting on a single GPU without any CPU offloading. It serves as a baseline to contrast with the offloaded training approaches in `strategies/naive_offload` and `strategies/clm_offload`.

## Gaussian Model (`./gaussian_model.py:GaussianModelNoOffload`)

This implements the original 3D Gaussian Splatting model from [gaussian-splatting/scene/gaussian_model.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py).

## Rendering Engine (`./engine.py`)

There are two main differences from the original 3D Gaussian Splatting rendering:

1. We use rendering kernels from [gsplat](https://github.com/nerfstudio-project/gsplat/blob/b60e917c95afc449c5be33a634f1f457e116ff5e/gsplat/rendering.py#L108).
2. We implement microbatch pipelining, rendering images one by one. 
