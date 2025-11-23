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

1. **Pre-allocation Capacity**: In `release_scripts/mip360.sh`, we have `--prealloc_capacity 7_000_000` to specify the number of Gaussians to pre-allocate in CPU pinned memory when using `--clm_offload` mode. If you have limited CPU memory (< 8GB), you may encounter CPU out-of-memory errors. In this case, you should reduce densification aggressiveness and lower the `--prealloc_capacity` value accordingly. 

2. **Batch Size**: We use `--bsz 4` for all scenes in these experiments. 

## Experimental Results and Performance Comparison

Below are the performance metrics (test PSNR, max GPU memory usage, and number of Gaussians) for all scenes across different offloading modes. 

These results were obtained on our testbed: AMD Ryzen Threadripper PRO 5955WX 16-core CPU, 128 GB RAM, and NVIDIA RTX 4090 GPU over PCIe 4.0.

### Test PSNR

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |         25.24 |           25.24 |        25.22 |
| bonsai  |         32.11 |           31.81 |        32.1  |
| counter |         29.07 |           29.08 |        28.97 |
| garden  |         27.36 |           27.35 |        27.31 |
| kitchen |         31.53 |           31.48 |        31.4  |
| room    |         31.39 |           31.45 |        31.46 |
| stump   |         26.7  |           26.68 |        26.62 |

### Maximum GPU Memory Usage (GB)

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |          3.01 |            4.8  |         8.21 |
| bonsai  |          0.82 |            1.1  |         1.75 |
| counter |          0.97 |            1.25 |         1.73 |
| garden  |          3.03 |            4.78 |         7.7  |
| kitchen |          1.54 |            2.08 |         2.49 |
| room    |          0.99 |            1.3  |         2.18 |
| stump   |          1.99 |            3.5  |         6.57 |

### Final Gaussian Count at the end of training

| Scene   |   CLM Offload |   Naive Offload |   No Offload |
|:--------|--------------:|----------------:|-------------:|
| bicycle |       6059752 |         6042092 |      5948420 |
| bonsai  |       1192793 |         1221540 |      1215377 |
| counter |       1186214 |         1184420 |      1190278 |
| garden  |       5559175 |         5555428 |      5575947 |
| kitchen |       1768105 |         1760823 |      1776933 |
| room    |       1543490 |         1541171 |      1551058 |
| stump   |       4729374 |         4873021 |      4752599 |