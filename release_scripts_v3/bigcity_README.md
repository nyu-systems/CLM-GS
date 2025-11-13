# CLM-GS Example Scripts: Matrixcity BigCity Aerial View

NOTE: you should launch these scripts in the base folder of this repository (where train.py is located). 
We call it `path/to/clm-gs`. 

## Download the matrixcity dataset

Download the dataset following instruction here: https://city-super.github.io/matrixcity/

There are small city and big city. In this script, we use big city. 

## Run the experiments

You need to specify the folder storing the mip360 dataset; and specify the mode for the training. 

This will run the scenes of counter bicycle stump garden room bonsai kitchen all together. 

```
CUDA_VISIBLE_DEVICES=1 bash release_scripts/bigcity.sh /path/to/360_v2 no_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/bigcity.sh /path/to/360_v2 clm_offload
CUDA_VISIBLE_DEVICES=1 bash release_scripts/bigcity.sh /path/to/360_v2 naive_offload
```

The training results will be saved into `path/to/clm-gs/output/bigcity`. 

After all experiment finishes, you can also run `python log2csv.py path/to/clm-gs/output/bigcity` 

## Experiment results in our testbed

We report the PSNR, training time, max GPU memory and CPU memory usage for the three modes. 
