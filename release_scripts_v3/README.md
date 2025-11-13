# CLM-GS Example Scripts

This folder contains example training scripts to help you understand and use CLM-GS codebase for large-scale 3D Gaussian Splatting reconstruction.

## Overview

We provide three types of example scripts, each demonstrating different aspects of our system:

1. **mip360.sh** - Standard quality benchmark on mip-NeRF 360 dataset
2. **rubble.sh** - Real-world large-scale scene reconstruction
3. **bigcity.sh** - Extreme-scale synthetic scene (capability upper bound)

## Offload Strategies

Our codebase supports three different memory management strategies:

### 1. `no_offload`
- **Description**: Keeps all Gaussian parameters and optimizer states on GPU
- **Use case**: Small to medium scenes that fit in GPU memory
- **Pros**: Fastest training speed
- **Cons**: Limited by GPU memory capacity

### 2. `naive_offload`
- **Description**: Offloads optimizer states to CPU, keeps Gaussians on GPU
- **Use case**: Medium to large scenes
- **Pros**: Simple offloading strategy, better memory efficiency than no_offload
- **Cons**: CPU-GPU transfer overhead

### 3. `clm_offload`
- **Description**: Compressed Learned Memory offloading with optimized data movement
- **Use case**: Large to extreme-scale scenes
- **Pros**: Highest memory efficiency, optimized for massive scenes
- **Cons**: Requires careful capacity planning

## Our testbed

One is a machine with an AMD Ryzen Threadripper PRO 5955WX 16-core CPU, 128 GB RAM, and a 24 GB NVIDIA RTX 4090 GPU connected over PCIe 4.0; 

## script to extract results from these log files

**log2csv.py**: Suppose `/path/to/folder` contains multiple experiment folders, each represents one experiment, you should use `python log2csv.py /path/to/folder` to 
list all of these experiments into a csv files for comparison.

## Explain some flags

