# CLM-GS Example Scripts: mip360

This file contains instructions for training mip360 datasets. 

Actually, mip360 is a small dataset and typically fits in GPU memory without the need for CPU offloading. These experiments are primarily to verify that PSNR results are consistent across different offloading modes. 

## Download the mip360 dataset

Download the dataset from https://jonbarron.info/mipnerf360/ .
You can simply run: wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

You should unzip the dataset into `/path/to/360_v2`. 

## Run the experiments

Run these below scripts from the root directory of the repository (the folder containing `train.py`, referred to here as `path/to/clm-gs`).

You need to specify the folder storing the mip360 dataset; and specify the mode for the training: 

```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 naive_offload
```

The `release_scripts/mip360.sh` script sequentially trains all seven mip360 scenes one after another: counter, bicycle, stump, garden, room, bonsai, and kitchen. 

The training results will be saved into `path/to/clm-gs/output/mip360`. 

Once all experiments have finished, you can the training statistics for all experiments by running:

```bash
python release_scripts/log2csv.py path/to/clm-gs/output/mip360
```

This command will process the experiment logs and produce a CSV of training results at `path/to/clm-gs/output/mip360/experiment_results.csv`.

## Notes on Hyperparameters

1. **Pre-allocation Capacity**: `--prealloc_capacity 7_000_000` specifies the number of Gaussians to pre-allocate in CPU pinned memory when using `--clm_offload` mode. If you have limited CPU memory (< 8GB), you may encounter out-of-memory errors. In this case, reduce densification aggressiveness and lower the `--prealloc_capacity` value accordingly. 

2. **Batch Size**: We use `--bsz 4` for all scenes in these experiments. 

## PSNR Comparison by Scene and Offload Type

Below are the PSNR results for all scenes using each offloading mode. These values were obtained on our testbed: AMD Ryzen Threadripper PRO 5955WX 16-core CPU, 128 GB RAM, and an NVIDIA RTX 4090 GPU over PCIe 4.0.

## Test PSNR

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |         25.24 |           25.24 |        25.22 |
| bonsai  |         32.11 |           31.81 |        32.1  |
| counter |         29.07 |           29.08 |        28.97 |
| garden  |         27.36 |           27.35 |        27.31 |
| kitchen |         31.53 |           31.48 |        31.4  |
| room    |         31.39 |           31.45 |        31.46 |
| stump   |         26.7  |           26.68 |        26.62 |

## Train PSNR

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |         26.38 |           26.37 |        26.23 |
| bonsai  |         33.26 |           33.26 |        33.43 |
| counter |         30.5  |           30.65 |        30.47 |
| garden  |         29.8  |           29.85 |        29.72 |
| kitchen |         33.15 |           32.93 |        32.77 |
| room    |         34.17 |           34.21 |        34.18 |
| stump   |         30.69 |           31.09 |        30.63 |
