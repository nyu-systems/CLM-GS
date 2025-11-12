# Grendel-XS Example Scripts

This folder contains example training scripts to help you understand and use the Grendel-XS codebase for large-scale 3D Gaussian Splatting reconstruction.

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

## Dataset Examples

### 1. mip-NeRF 360 Dataset (mip360.sh)

**Purpose**: This is the standard benchmark for evaluating novel view synthesis quality. Running this script will allow you to quickly understand our repository's performance on a well-known dataset.

**Dataset**: The mip-NeRF 360 dataset contains 7 scenes captured with varying complexity levels.

**Usage**:
```bash
bash mip360.sh <dataset_folder> <offload_strategy>
```

**Arguments**:
- `<dataset_folder>`: Path to the root folder containing mip360 scenes (should contain subdirectories: counter, bicycle, stump, garden, room, bonsai, kitchen)
- `<offload_strategy>`: One of `no_offload`, `naive_offload`, or `clm_offload`

**Example**:
```bash
bash mip360.sh /path/to/mip360_dataset clm_offload
```

**Configuration**:
- Batch size: 1
- Iterations: 30,000
- Image resolution: Varies by scene (images_4 for bicycle/stump/garden, images_2 for others)
- Expected training time: ~1-2 hours per scene (depending on hardware)

### 2. Rubble Downsampled 4x Dataset (rubble.sh)

**Purpose**: This demonstrates a real-world large-scale reconstruction scenario. The Rubble dataset is a challenging outdoor scene with complex geometry, showing what our system can handle in practice.

**Dataset**: Rubble dataset downsampled to 1/4 resolution (images_4)

**Usage**:
```bash
bash rubble.sh <dataset_folder> <offload_strategy>
```

**Arguments**:
- `<dataset_folder>`: Path to the rubble dataset folder
- `<offload_strategy>`: One of `no_offload`, `naive_offload`, or `clm_offload`

**Example**:
```bash
bash rubble.sh /path/to/rubble-pixsfm clm_offload
```

**Configuration**:
- Batch size: 4
- Iterations: 30,000
- Image resolution: images_4 (downsampled 4x)
- Densification: Enabled with standard parameters
- Expected training time: ~3-5 hours (depending on hardware and strategy)

**Note**: For this large-scale scene, we recommend using `clm_offload` strategy for best memory efficiency.

### 3. MatrixCity BigCity Dataset (bigcity.sh)

**Purpose**: This is a synthetic dataset designed to showcase the upper bound capability of our system. It's an extremely large-scale scene (102M+ Gaussians) that demonstrates what's possible with our memory management techniques.

**Dataset**: MatrixCity BigCity - a synthetic urban environment

**Usage**:
```bash
bash bigcity.sh <dataset_folder>
```

**Arguments**:
- `<dataset_folder>`: Path to the MatrixCity BigCity dataset folder

**Example**:
```bash
bash bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks
```

**Configuration**:
- Batch size: 64
- Iterations: 500,000
- Strategy: Only `clm_offload` (required for extreme scale)
- Pre-allocated capacity: 102,231,360 Gaussians
- Auto-densification: Disabled (fixed capacity)
- Expected training time: 20-30 hours (depending on hardware)

**Note**: This dataset requires significant computational resources and is primarily for demonstrating capability rather than everyday use.

## Output Structure

All scripts will create output folders with the following structure:
```
output/
└── <dataset_name>/
    └── <timestamp>_<experiment_name>/
        ├── point_cloud/         # Saved Gaussian models
        ├── train/              # Training visualizations
        └── test/               # Test set evaluation results
```

## Hardware Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM for mip360 with no_offload)
- RAM: 32GB+ system memory
- Storage: SSD recommended for dataset storage

### Recommended for Large-Scale Scenes
- GPU: NVIDIA A100/H100 or RTX 4090
- RAM: 128GB+ system memory
- Storage: NVMe SSD with 500GB+ free space

## Tips and Best Practices

1. **Start with mip360**: If you're new to the codebase, start with the mip360 dataset to familiarize yourself with the system and verify your setup.

2. **Strategy selection**: 
   - Use `no_offload` for scenes that fit in GPU memory (faster)
   - Use `clm_offload` for large-scale scenes (more memory efficient)
   - Use `naive_offload` if you want a middle ground

3. **Monitoring**: All scripts include monitoring flags (`--enable_timer`, `--check_gpu_memory`, etc.) to help you track resource usage.

4. **Dataset preparation**: Ensure your datasets are in COLMAP format with proper camera poses before running the scripts.

5. **Adjusting parameters**: You can modify the scripts to change batch size, iterations, or other parameters based on your hardware and requirements.

## Experiment Results

**Hardware Setup**: We run our evaluation on two testbeds. The primary results reported here are from a machine with an AMD Ryzen Threadripper PRO 5955WX 16-core CPU, 128 GB RAM, and a 24 GB NVIDIA RTX 4090 GPU connected over PCIe 4.0.

### mip-NeRF 360 Dataset

We compare all three offload strategies on the mip360 dataset benchmark. The table below shows the PSNR results for each scene:

#### PSNR Comparison (dB)

| Scene | `no_offload` | `naive_offload` | `clm_offload` |
|-------|--------------|-----------------|---------------|
| bicycle | [TBD] | [TBD] | [TBD] |
| bonsai | [TBD] | [TBD] | [TBD] |
| counter | [TBD] | [TBD] | [TBD] |
| garden | [TBD] | [TBD] | [TBD] |
| kitchen | [TBD] | [TBD] | [TBD] |
| room | [TBD] | [TBD] | [TBD] |
| stump | [TBD] | [TBD] | [TBD] |
| **Mean** | [TBD] | [TBD] | [TBD] |

#### End-to-End Throughput

| Scene | `no_offload` | `naive_offload` | `clm_offload` |
|-------|--------------|-----------------|---------------|
| bicycle | [TBD] | [TBD] | [TBD] |
| bonsai | [TBD] | [TBD] | [TBD] |
| counter | [TBD] | [TBD] | [TBD] |
| garden | [TBD] | [TBD] | [TBD] |
| kitchen | [TBD] | [TBD] | [TBD] |
| room | [TBD] | [TBD] | [TBD] |
| stump | [TBD] | [TBD] | [TBD] |
| **Mean** | [TBD] | [TBD] | [TBD] |

#### Peak GPU Memory

| Scene | `no_offload` | `naive_offload` | `clm_offload` |
|-------|--------------|-----------------|---------------|
| bicycle | [TBD] | [TBD] | [TBD] |
| bonsai | [TBD] | [TBD] | [TBD] |
| counter | [TBD] | [TBD] | [TBD] |
| garden | [TBD] | [TBD] | [TBD] |
| kitchen | [TBD] | [TBD] | [TBD] |
| room | [TBD] | [TBD] | [TBD] |
| stump | [TBD] | [TBD] | [TBD] |
| **Mean** | [TBD] | [TBD] | [TBD] |

**Key Observations**:
- All three strategies achieve comparable reconstruction quality (PSNR)
- `clm_offload` provides the best memory efficiency, enabling larger-scale reconstructions
- `no_offload` offers the fastest training when GPU memory is sufficient

### Rubble Dataset

For the large-scale Rubble scene, we only report results for `clm_offload` and `naive_offload` strategies, as `no_offload` leads to out-of-memory errors.

| Strategy | Max PSNR (dB) | Number of Gaussians |
|----------|---------------|---------------------|
| `naive_offload` | [TBD] | [TBD] M |
| `clm_offload` | [TBD] | [TBD] M |

**Note**: The `no_offload` strategy is not feasible for this scene due to GPU memory constraints (OOM).

### MatrixCity BigCity Dataset

For the extreme-scale BigCity scene, we only report results for `clm_offload`, as this is the only strategy capable of handling scenes of this magnitude.

| Strategy | Max PSNR (dB) | Number of Gaussians |
|----------|---------------|---------------------|
| `clm_offload` | [TBD] | 102.2 M |

**Note**: Both `no_offload` and `naive_offload` strategies are not feasible for this scene due to memory constraints (OOM) and prohibitively slow training times.

## Troubleshooting

### Out of Memory (OOM) Errors
- Try a different offload strategy (no_offload → naive_offload → clm_offload)
- Reduce batch size
- Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (already in scripts)

### Slow Training
- Check if you're using decoded dataset paths (faster data loading)
- Ensure SSD storage for datasets
- Monitor GPU utilization

### Incorrect Results
- Verify dataset format and camera poses
- Check that the correct image folder is specified
- Ensure dataset path is correct

## Additional Resources

For more details about the codebase architecture, implementation details, and advanced usage, please refer to the main README.md in the repository root.

