<p align="center">
    <!-- license badge -->
    <a href="https://github.com/nerfstudio-project/nerfstudio/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <!-- stars badge -->
    <a href="https://github.com/nyu-systems/CLM-GS/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/nyu-systems/CLM-GS?style=social"/>
    </a>
    <!-- pull requests badge -->
    <a href="https://github.com/nyu-systems/CLM-GS/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/nyu-systems/CLM-GS"/>
    </a>

</p>


<div align="center">

CLM-GS
===========================
_<h4>Removing the GPU Memory Barrier for 3D Gaussian Splatting with CPU Offloading</h4>_
_<h4>✨ Accepted to appear in ASPLOS 2026 ✨</h4>_

### [Paper](https://arxiv.org/abs/2511.04951) | [Project Page](https://tarzanzhao.github.io/CLM-GS/)

<div align="left">

<div align="center">
    <img src="assets/teaser_11210004_2x.gif" width="1000">
</div>

# Overview

CLM enables training of large-scale 3D Gaussian Splatting scenes that exceed GPU memory capacity by also exploiting CPU memory. 

By using CLM offloading, your 3DGS training can:
- **Train large-scale scenes** with 100+ million Gaussians on a single 24GB GPU and 128GB RAM
- **Maintain rendering quality** with a mathematically identical rendering formula
- **Work with existing rendering kernels**: We use off-the-shelf rendering kernels from gsplat. Our offloading design is orthogonal to these rendering kernels, making it easy to integrate with your own splatting pipelines

This codebase provides three modes of memory-efficient training strategies for your reference:
- **no_offload**: 3DGS training only on GPU. We optimize memory usage with engineering tricks on a single GPU. The show that CLM's offloading does not affect quality. (Implemented in `strategies/no_offload`)
- **naive_offload**: A simple CPU offloading implementation that stores all Gaussian attributes (xyz, etc.) and their optimizer states on CPU, loads parameters onto GPU in each iteration, and offloads gradients back to CPU in each batch. This demonstrates the simplest offloading strategy, though it is slower. (Implemented in `strategies/naive_offload`)
- **clm_offload**: Our most sophisticated offloading design that keeps selection-critical attributes on GPU while offloading others to CPU along with their optimizer states. It reduces memory usage to the extreme while maintaining good speed. The code is more complex but highly efficient. (Implemented in  `strategies/clm_offload`) 

**Table of contents**
-----
- [Why use CLM-GS?](#why-use-clm-gs)
- [How to use CLM?](#how-to-use-clm-gs)
    - [Setup](#setup)
    - [Training](#training)
    - [Consideration about the flags](#considerations-about-the-flags)
- [Example Usages](#example-usages)
- [Paper](#paper-and-citation)
- [License](#license)
- [Reference](#reference)
------
<!-- - [Implementation Details](#implementation-details) -->

# Why use CLM-GS

**The goal of CLM-GS is to solve GPU out-of-memory problems in 3DGS Training.**

Traditional 3D Gaussian Splatting stores all parameters, optimizer states, and activation states on GPU, which severely limits the scene scale you can reconstruct due to GPU memory constraints (24GB on 4090). When the scene is very large and intricate, the large number of required Gaussians linearly increases memory consumption for parameters and optimizer states. When rendering high-resolution images, activation states also grow larger. As a result, GPU out-of-memory errors become a common issue.

CLM-GS addresses these memory constraints effectively. The table below compares GPU memory usage and training time across different scenes (102M means 102 million Gaussians) on our RTX 4090 testbed:

| Strategy        | Bicycle (6M)           | Rubble 4K (10M)        | Rubble 4K (28M)        | BigCity Aerial (102M)  |
|:----------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| `no_offload`    | 8.21 GB / 734 s          | 16.81 GB / 11702 s     | OOM                    | OOM                    |
| `naive_offload` | 4.80 GB / 2481 s          | 9.32 GB / 22254 s      | 19.03 GB / 40820 s     | OOM                    |
| `clm_offload`   | 3.01 GB / 1348 s          | 7.05 GB / 12381 s      | 13.0 GB / 24757 s      | 20.79 GB / 11783 s     |


# How to use CLM-GS

## Setup

### Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
git clone git@github.com:nyu-systems/CLM-GS.git --recursive
```

### A Conda Environment

Ensure you have Conda, GPU with compatible driver and CUDA environment installed on your machine, as prerequisites.

**Note**: PyTorch version >= 2.6 is required due to the usage of `torch.nonzero_static()` API.

Create and activate the conda environment:
```shell
conda create -n clm_gs python=3.10
conda activate clm_gs
```

Install PyTorch and related packages (please install a compatible Python and PyTorch set of packages for your system), for example:
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install additional dependencies:
```shell
pip install tqdm plyfile psutil numba opencv-python scipy matplotlib pandas imageio imageio-ffmpeg requests tabulate
```

Compile and Install submodules locally:
```shell
pip install --no-build-isolation submodules/clm_kernels
pip install --no-build-isolation submodules/cpu-adam
pip install submodules/fast-tsp
pip install --no-build-isolation submodules/gsplat
pip install --no-build-isolation submodules/simple-knn
```

## Training

### Dataset Preparation

This repository trains a 3D Gaussian Splatting model using COLMAP-formatted input datasets. A COLMAP-formatted dataset contains a list of images with their corresponding camera poses, as well as an initial sparse point cloud that roughly represents the scene structure. This repository can reconstruct a detailed 3DGS model that captures intricate details from these images within the colmap-formatted dataset.

The following two COLMAP-formatted example datasets are available for use in the following guide:
- **Mip360 Dataset**: Download from https://jonbarron.info/mipnerf360/
- **Rubble 4K Dataset**: Download from https://huggingface.co/datasets/HexuZhao/mega_nerf_rubble_colmap/tree/main

<!-- 
See [Tutorial 1](release_scripts/mip360_README.md) and [Tutorial 2](release_scripts/rubble4k_README.md) for detailed instructions on training with these datasets to achieve optimal performance.  -->

### Basic Training with Different Strategies

**No Offload (GPU-Only, for small scenes)**:
```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py -s <path to COLMAP dataset> --no_offload --bsz 4
```

**Naive Offload (Simple offloading for medium scenes)**:
```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py -s <path to COLMAP dataset> --naive_offload --bsz 4
```

**CLM Offload (Recommended for large scenes)**:
```shell
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py -s <path to COLMAP dataset> --clm_offload --bsz 4
```

## Considerations about the flags

### **Three offloading strategies**

To be simple, `--no_offload` is just a GPU-only training baseline for the other two offloading strategies to compare. 
And the `--naive_offload` is an easy implementation but it is slow and cannot handle extreme large scene; `--clm_offload` is fast and can support even larger Gaussians model. 

For detailed experimental setups and performance comparisons, see the "Why use CLM-GS?" section above and the `Example Usages` section below.

### **Dataset Caching and Streaming**

This codebase saves the dataset on disk and loads it on-demand during training to conserve CPU RAM. This is because, for extremely large datasets, you may not be able to decode the entire dataset into CPU RAM, let alone GPU memory. Note that streaming from disk to GPU is slower than streaming from CPU RAM to GPU.

Use `--dataset_cache_and_stream_mode` to control how images are handled:

**Mode 1: "decode_images_in_advance" (Default)**

This mode decodes JPG/PNG images into raw byte data when you first train on a dataset at a specific resolution, allowing on-demand streaming during training.

- **Storage Location**: Ensure decoded images are saved on a local disk rather than a network file system (NFS). Loading from NFS is significantly slower. The default decoded path is `--source_path/decode_{args.images}`. If `--source_path` is on an NFS, specify `--decode_dataset_path` to point to a local disk location.
- **Disk Space**: The decoded dataset can be very large. Calculate the required space as:
  ```
  Disk Space = (num_images × image_height × image_width × 3) bytes
  ```
- **First-Time Setup**: Initial decoding takes time, but the decoded dataset can be reused for subsequent training runs on the same scene.

If the decoded images path is corrupted or missing, simply remove the folder and rerun the decoding process.

**Mode 2: "decode_images_on_demand"**

This mode avoids pre-decoding images, saving disk storage space. However, decoding images on the CPU before each rendering pass is slower and consumes additional CPU computation.

### **Pre-allocate buffers for Gaussians on CPU RAM**

For CLM offload mode, you can specify how many Gaussians to pre-allocate in CPU pinned memory:

```shell
python train.py -s <path to COLMAP dataset> \
    --clm_offload \
    --prealloc_capacity 40000000  # Pre-allocate for 40M Gaussians
```

If you don't specify `--prealloc_capacity`, the system automatically calculates the maximum number of Gaussians your workstation can support:

```
Number of Gaussians = (remaining CPU memory × 0.7) / (48 × 4 × 4 bytes)
```

**Rule of thumb**: Approximately 8 GB CPU memory per 10 million Gaussians.

Where:
- 48 = number of spherical harmonic coefficients offloaded to CPU RAM
- First 4 = bytes per float32
- Second 4 = storage multiplier (parameter + gradient + 2 optimizer states)
- 0.7 is a conservative multiplier that reserves memory for other workloads. For more aggressive memory allocation, specify `--prealloc_capacity` explicitly.

**Note**: `--prealloc_capacity` is only effective when `--clm_offload` is enabled.

#### **Microbatch Pipelining**

This codebase uses microbatch pipelining with gradient accumulation. For each microbatch, we render one image and perform one backpropagation. The `--bsz` flag controls how many images to process before each optimizer step.

This design choice is important. Without microbatch pipelining, activation memory would grow linearly with batch size. With pipelining, activation memory remains constant at the level needed for rendering a single image.

Learning rate and momentum are scaled according to Grendel-GS rules when increasing `--bsz`. Currently, `--clm_offload` supports batch sizes of 4, 8, 16, 32, and 64. 

<details>
<summary><span style="font-weight: bold;">CLM-specific Command Line Arguments for train.py</span></summary>

  #### --no_offload
  Use GPU-only mode (no parameter offloading). Best for small scenes.
  
  #### --naive_offload
  Use simple offloading strategy. Suitable for medium scenes.
  
  #### --clm_offload
  Use CLM offloading with retention optimization. Best for large scenes.
  
  #### --prealloc_capacity
  Number of Gaussians to pre-allocate in CPU pinned memory (e.g., `40000000` for 40M). Required for CLM offload with densification.
 
  #### --bsz
  Batch size using micro-batch pipelining with gradient accumulation. `--bsz 4` renders and backpropagates 4 images sequentially before each optimizer step. Images are processed one-by-one rather than simultaneously to reduce activation memory usage. 

  #### All other arguments
  Please follow Gaussian Splatting's original codebase. 

</details>
<br>

---

# Example Usages

This section demonstrates CLM-GS on three different scales of scenes, from small benchmarks to extreme-scale reconstructions. Each example includes detailed reproduction instructions and usage pipelines. 
In all examples, we set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation in PyTorch's CUDA memory allocator. 

## Our Testbed

All experiments were conducted on:
- **Hardware**: AMD Ryzen Threadripper PRO 5955WX (16-core), 128GB RAM, NVIDIA RTX 4090 24GB
- **Interconnect**: PCIe 4.0 x16 (CPU-GPU)

## Example 1: Mip-NeRF 360 Dataset (Small-Scale Benchmark)

The Mip-NeRF 360 dataset provides standard benchmark scenes for evaluating quality and performance. While these scenes are small enough to fit in GPU memory, they serve as a baseline to verify that CLM offloading maintains quality while reducing memory usage.

📖 **[Complete Mip360 Tutorial](release_scripts/mip360_README.md)**

---

## Example 2: Rubble 4K Dataset (Large-Scale Reconstruction)

The MegaNeRF Rubble scene at 4K resolution represents a real-world large-scale outdoor scene that exceeds standard GPU memory capacity. This example demonstrates CLM's ability to train a real-world large-scale scene from scratch. 

📖 **[Complete Rubble 4K Tutorial](release_scripts/rubble4k_README.md)**

---

## Example 3: MatrixCity BigCity Dataset (Extreme-Scale Reconstruction)

The MatrixCity BigCity dataset represents the extreme upper bound of scene reconstruction with synthetic city-scale environments. This demonstrates CLM's capability to handle 100 million Gaussians. This serves as a stress test, requiring 128GB RAM and 24GB GPU memory to successfully train with 100 million Gaussians. 

📖 **[Complete BigCity Tutorial](release_scripts/bigcity_README.md)**

---

<!-- # Implementation Details

This section covers engineering-level optimizations implemented in CLM-GS for efficient memory management:

## Memory-Efficient Optimizations

CLM-GS includes several low-level optimizations to maximize memory efficiency:

- **Disk-based dataset streaming**: The dataset is stored on disk and streamed on-demand to reduce both RAM and GPU memory consumption. This is particularly important for extremely large datasets that cannot be fully decoded into CPU RAM.

- **Aggressive memory management**: The codebase frequently releases memory and reduces memory fragmentation throughout the training process.

- **PyTorch memory allocation**: Setting `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces memory fragmentation in PyTorch's CUDA memory allocator. This environment variable is included in all training commands in the examples above.

These optimizations work in conjunction with the three offloading strategies to achieve maximum memory efficiency across different hardware configurations. -->

---

# Paper and Citation

Our system design, memory management strategies, and scaling insights are documented in the paper below: 

> [**CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting**](https://arxiv.org/abs/2511.04951)<br>
> [**Hexu Zhao¹\***](http://tarzanzhao.github.io/), [**Xiwen Min¹\***](https://github.com/alexis-mmm), [**Xiaoteng Liu¹**](https://www.linkedin.com/in/xiaoteng-frank-liu-95277b232/), [**Moonjun Gong¹**](https://moonjungong.github.io), [**Yiming Li¹**](https://yimingli-page.github.io), [**Ang Li²,³**](https://www.angliphd.com), [**Saining Xie¹**](https://www.sainingxie.com), [**Jinyang Li¹**](https://jinyangli.github.io), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/)  (\* *co-first authors*)
> <br>¹New York University, ²Pacific Northwest National Laboratory, ³University of Washington <br>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{zhao2025clm,
      title={CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting},
      author={Hexu Zhao and Xiwen Min and Xiaoteng Liu and Moonjun Gong and Yiming Li and Ang Li and Saining Xie and Jinyang Li and Aurojit Panda},
      booktitle={Proceedings of the 2026 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'26)},
      year={2026},
      address={Pittsburgh, PA, USA},
      url={https://arxiv.org/abs/2511.04951}
    }</code></pre>
  </div>
</section> 

# Contributing

Please use [Black](https://github.com/psf/black) with default settings to format code.

```shell
# Install
pip install black
# Format all files
black .  
```

# License

Distributed under the Apache License Version 2.0 License. See `LICENSE.txt` for more information.

# Reference

1. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, July 2023. URL: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
