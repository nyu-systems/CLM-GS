# CLM-GS Example Scripts: MatrixCity BigCity Aerial Dataset

This script shows how to train the MatrixCity BigCity Aerial dataset using the CLM system at extreme scale, with either 102 million or 25 million Gaussians. This serves as a full stress test for CLM.

In this example, the training starts from a very large initial point cloud (102 million 3D points). Densification is disabled during training because the default 3DGS densification is not effective for such large-scale scenes (or the hyperparameters are just too hard to tune). 

## Download the MatrixCity BigCity Aerial Dataset

MatrixCity has both SmallCity and BigCity, and both Aerial view and Street view datasets. This example script only uses the BigCity Aerial view, so you only need to download that portion.

You can download by following official instructions from MatrixCity (https://city-super.github.io/matrixcity/). Specifically, you need to download the BigCity Aerial dataset from here (using their Hugging Face repository as an example): https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/big_city. After downloading, decompress the zip files in `path/to/big_city/aerial/test` and `path/to/big_city/aerial/train`. 

Next, you need to download some additional files provided by us: https://huggingface.co/datasets/HexuZhao/matrixcity_bigcity_aerial_102M_initial_point_cloud

Place the `all_blocks` folder downloaded from this link at `path/to/big_city/aerial/pose/all_blocks` (you may need to create pose folder first). This folder contains:
1. JSON files with camera poses for different images
2. An initial point cloud containing 102 million points

## Expected Dataset Folder Structure

```
/path/to/matrixcity/
└── big_city/
    └── aerial/
        ├── train/
        ├── test/
        └── pose/
            └── all_blocks/
                ├── transforms_train.json
                ├── transforms_test.json
                ├── transforms_train_ocean_info.json
                ├── transforms_test_ocean_info.json
                └── all_ds32_102231360.ply         # 102M or 25M point cloud file
```

Where:
- `/path/to/matrixcity/big_city/aerial/{train, test}` contains the official MatrixCity BigCity Aerial dataset files (decompressed from their Hugging Face repository).
- `/path/to/matrixcity/big_city/aerial/pose/all_blocks/` contains the JSON files and the large initial point cloud (downloaded from the `HexuZhao/matrixcity_bigcity_aerial_102M_initial_point_cloud` Hugging Face link).

**Important:** Ensure your folders are organized exactly as shown above. This is necessary for the training scripts to locate the images correctly. 

## Running Experiments

Run these below scripts from the root directory of the repository (the folder containing `train.py`, referred to here as `path/to/clm-gs`).

The script accepts three parameters:
1. **dataset_folder**: Path to the MatrixCity BigCity dataset
2. **offload_strategy**: One of `no_offload`, `clm_offload`, or `naive_offload`
3. **scale**: Either `102m` (100 million Gaussians) or `25m` (25 million Gaussians)

### Command Syntax

```bash
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh <dataset_folder> <offload_strategy> <scale>
```

### Example Commands

```bash
# 100M Gaussians experiments
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks no_offload 102m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 102m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks naive_offload 102m

# 25M Gaussians experiments
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks no_offload 25m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks clm_offload 25m
CUDA_VISIBLE_DEVICES=0 bash release_scripts/bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks naive_offload 25m
```

<!-- CUDA_VISIBLE_DEVICES=1 bash release_scripts/bigcity.sh /mnt/nvme0/dataset/matrixcity/big_city/aerial/pose/all_blocks 25m -->

**Note:** `no_offload` and `naive_offload` strategies typically result in out-of-memory (OOM) errors for 100M scale. Only `clm_offload` can successfully handle extreme scales.

Training results will be saved to `output/bigcity/<timestamp>_bigcity_<scale>_<strategy>/`.

### Analyzing Results

After all experiments finish, you can aggregate results using:

```bash
python release_scripts/log2csv.py output/bigcity
```

This will generate a CSV summary of PSNR, training time, and memory usage across all experiments.

## Notes on Hyperparameters

### 102M and 25M Gaussians

The 102M scale uses the 102M initial point cloud. The 25M version is created by randomly sampling 25% of the 102M initial point cloud using the `--initial_point_cloud_downsampled_ratio 0.25` flag. 

### Pre-allocated Capacity

For `clm_offload`, the script pre-allocates 102 Millions gaussians capacity in CPU pinned memory. If your system has small CPU memory (e.g., < 100GB), you may be only able to train the 25M scale setup which uses less aggressive densification. 

### Checkpoints Saving

We use `--save_tensors` in these scripts, which saves `.pt` files instead of `.ply` files for checkpoints because `.pt` is faster and consumes less CPU memory. This is a necessary flag. 

### Training steps

We train for 500000 images times in this `bigcity.sh` script. 

## Experiments Results on our testbed

Below are the performance metrics (PSNR, max GPU memory usage, and number of Gaussians etc) across different offloading modes and model sizes. 

The `clm_offload` strategy demonstrates significant memory efficiency while maintaining rendering quality, enabling training at scales that would otherwise be impossible on standard hardware. 

| Experiment                    | Test PSNR   | Train PSNR   | Num 3DGS   | Max GPU Memory (GB)   | Pinned CPU Memory (GB)   | Training Time (s)   |
|:------------------------------|:------------|:-------------|:-----------|:----------------------|:-------------------------|:--------------------|
| bigcity_102m_clm_offload_102M | 25.5        | 26.84        | 102231360  | 20.79                 | 37.41                    | 11783.36            |
| bigcity_102M_naive_offload    | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
| bigcity_102M_no_offload       | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |
| bigcity_25m_clm_offload_25M   | 24.63       | 25.9         | 25557840   | 5.64                  | 10.04                    | 6029.05             |
| bigcity_25m_naive_offload_25M | 24.29       | 25.12        | 25557840   | 12.13                 | 19.87                    | 10187.07            |
| bigcity_25M_no_offload        | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |

## Reference

