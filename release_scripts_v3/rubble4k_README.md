# CLM-GS Example Scripts: mip360

NOTE: you should launch these scripts in the base folder of this repository (where train.py is located). 
We call it `path/to/clm-gs`. 

## Download the rubble dataset

You can download the Rubble dataset from here: https://github.com/cmusatyalab/mega-nerf

We also store the dataset by ourself and you can download here: TBD

## Run the experiments

You need to specify the folder storing the mip360 dataset; and specify the mode for the training. 

This will run the scenes of counter bicycle stump garden room bonsai kitchen all together. 

```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /path/to/360_v2 naive_offload
```

The training results will be saved into `path/to/clm-gs/output/mip360`. 

After all experiment finishes, you can also run `python log2csv.py path/to/clm-gs/output/mip360` 

<!-- ```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/mip360.sh /mnt/nvme0/dataset/360_v2_20251111 naive_offload
``` -->

## Experiment results in our testbed

We report the PSNR, training time, max GPU memory and CPU memory usage for the three modes. 


## Notes on Hyperparameters

1. In this set of experiment, --prealloc_capacity 7_000_000 is the number of pre-allocated buffer on CPU pinned memory for storing gaussian. If your CPU memory is very very small (less than 8GB), you may encounter OOM. Then you need to make the densification less aggressive, and at the same time reduce the prealloc_capacity. 

