# CLM-GS Example Scripts: Rubble 4K

This file contains instructions for training Rubble 4k dataset. 

This demonstrates large-scale real-world scene reconstruction using the Rubble dataset. 

## Download the Rubble Dataset

The Rubble dataset is available from the Mega-NeRF project: https://github.com/cmusatyalab/mega-nerf. Note that the original dataset is not in COLMAP format. So you may need to run colmap on these rubble dataset images. 

For convenience, we provide a COLMAP-compatible version of the Rubble 4K dataset on Hugging Face (recommended): https://huggingface.co/datasets/HexuZhao/mega_nerf_rubble_colmap/tree/main

After downloading `rubble-colmap.zip` from this Hugging Face repository, extract it with:

```bash
unzip rubble-colmap.zip
```

This will create the folder `path/to/rubble-colmap` containing the dataset in COLMAP format.

## Running Experiments

Run these below scripts from the root directory of the repository (the folder containing `train.py`, referred to here as `path/to/clm-gs`).

The script accepts three parameters:
1. **dataset_folder**: Path to the Rubble dataset `path/to/rubble-colmap`.
2. **offload_strategy**: One of `no_offload`, `clm_offload`, or `naive_offload`. 
3. **scale**: Either `10m` (10 million Gaussians) or `28m` (28 million Gaussians). 

The scale parameter controls the aggressiveness of the densification process, resulting in different numbers of Gaussians. 

### Command Syntax

```bash
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh <dataset_folder> <offload_strategy> <scale>
```

### Example Commands

```bash
# 10M Gaussians experiments (less aggressive densification)
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap no_offload 10m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap clm_offload 10m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap naive_offload 10m

# 28M Gaussians experiments (more aggressive densification)
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap no_offload 28m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap clm_offload 28m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/rubble4k.sh /path/to/rubble-colmap naive_offload 28m
```

**Note:** The `clm_offload` strategy is recommended for memory efficiency, especially for the 28M scale.

Training results will be saved to `output/rubble/<timestamp>_rubble4k_<scale>_<strategy>/`.

### Analyzing Results

After all experiments finish, you can aggregate results using:

```bash
python release_scripts/log2csv.py output/rubble
```

This will generate a CSV summary of PSNR, training time, and memory usage etc across all experiments.

## Notes on Hyperparameters

### Densification Settings

- **10m: 10 Millions Gaussians**: Uses more conservative densification parameters
  - Gradient threshold: 0.00015
  - Percent dense: 0.005
  - Opacity reset interval: 3000
  - Densify from iteration 4000 to 21000

- **28m: 28 Millions Gaussians**: Uses more aggressive densification parameters
  - Gradient threshold: 0.0001
  - Percent dense: 0.002
  - Opacity reset interval: 9000
  - Densify from iteration 5000 to 21000

### Pre-allocated Capacity

For `clm_offload`, the script pre-allocates 32M capacity in CPU pinned memory. If your system has small CPU memory (e.g., < 32GB), you may need to:
1. Use the 10M scale which uses less aggressive densification
2. Then reduce the `--prealloc_capacity` parameter in the script

### Checkpoints Saving

We use `--save_tensors` in these scripts, which saves `.pt` files instead of `.ply` files for checkpoints because `.pt` is faster and consumes less CPU memory. 

### Training steps

We train for 100000 images times in this `rubble4k.sh` script. 

## Rubble 4K Experiment Results

Below are the performance metrics (PSNR, max GPU memory usage, and number of Gaussians etc) across different offloading modes and model sizes. 

These results were obtained on our testbed: AMD Ryzen Threadripper PRO 5955WX 16-core CPU, 128 GB RAM, and NVIDIA RTX 4090 GPU over PCIe 4.0.

Note: The rubble4k_28M_no_offload experiment runs out of memory (OOM) because a single GPU cannot train a model with 28 million Gaussians.

| Experiment                 | Test PSNR   | Train PSNR   | Num 3DGS   | Max GPU Memory (GB)   | Pinned CPU Memory (GB)   | Training Time (s)   |
|:---------------------------|:------------|:-------------|:-----------|:----------------------|:-------------------------|:--------------------|
| rubble4k_10m_clm_offload   | 26.03       | 27.4         | 10149035   | 7.05                  | 11.47                    | 12381.47            |
| rubble4k_10m_naive_offload | 25.92       | 27.29        | 10335575   | 9.32                  | 11.46                    | 22254.41            |
| rubble4k_10m_no_offload    | 26.14       | 27.36        | 10058114   | 16.81                 | 0.62                     | 11702.31            |
| rubble4k_28m_clm_offload   | 26.75       | 28.3         | 27992096   | 13.0                  | 12.32                    | 24757.44            |
| rubble4k_28m_naive_offload | 26.72       | 28.19        | 27385268   | 19.03                 | 14.58                    | 40820.35            |
| rubble4k_28M_no_offload    | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
