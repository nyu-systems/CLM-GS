#!/bin/bash

# rubble4k.sh - Training script for Rubble dataset
# This script demonstrates large-scale real-world scene reconstruction

# Check arguments
if [ $# -ne 3 ]; then
    echo "Error: Please specify exactly three arguments."
    echo "Usage: bash rubble4k.sh <dataset_folder> <offload_strategy> <scale>"
    echo ""
    echo "Arguments:"
    echo "  <dataset_folder>   : Path to the rubble dataset folder"
    echo "  <offload_strategy> : One of: no_offload, clm_offload, naive_offload"
    echo "  <scale>            : One of: 10m (10M Gaussians), 28m (28M Gaussians)"
    echo ""
    echo "Examples:"
    echo "  bash rubble4k.sh /path/to/rubble-pixsfm clm_offload 10m"
    echo "  bash rubble4k.sh /path/to/rubble-pixsfm clm_offload 28m"
    echo "  bash rubble4k.sh /path/to/rubble-pixsfm no_offload 10m"
    echo ""
    echo "Note: clm_offload is recommended for memory efficiency"
    exit 1
fi

dataset_folder=$1
offload_strategy=$2
scale=$3

# Validate offload strategy
if [[ ! "$offload_strategy" =~ ^(no_offload|clm_offload|naive_offload)$ ]]; then
    echo "Error: Invalid offload strategy '$offload_strategy'"
    echo "Must be one of: no_offload, clm_offload, naive_offload"
    exit 1
fi

# Validate scale
if [[ ! "$scale" =~ ^(10m|28m)$ ]]; then
    echo "Error: Invalid scale '$scale'"
    echo "Must be one of: 10m, 28m"
    exit 1
fi

echo "Dataset folder: $dataset_folder"
echo "Offload strategy: $offload_strategy"
echo "Scale: $scale"

# Generate timestamp for experiment naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Experiment name
expe_name="rubble2k_${scale}_${offload_strategy}"

# Set output paths
log_folder="output/rubble2k/${TIMESTAMP}_${expe_name}"
model_path=${log_folder}

echo "Output folder: $log_folder"

DOWNSAMPLE_OPT="--images images_2"

# Training configurations
LLFFHOLD=83
BSZ=4
ITERATIONS=100000
# ITERATIONS=7000
LOG_INTERVAL=250

# Test and save iterations
TEST_ITERATIONS="7000 20000 40000 60000 80000 100000"
SAVE_ITERATIONS="40000 100000"
# SAVE_ITERATIONS=""

# Set densification parameters based on scale
if [ "$scale" == "10m" ]; then
    # 10M scale: More conservative densification parameters
    DENSIFY_OPTS="--densify_from_iter 4000 \
--densify_until_iter 21000 \
--densify_grad_threshold 0.00015 \
--percent_dense 0.005 \
--opacity_reset_interval 3000"
    SCALE_DESC="10M"
elif [ "$scale" == "28m" ]; then
    # 28M scale: More aggressive densification parameters
    DENSIFY_OPTS="--densify_from_iter 5000 \
--densify_until_iter 21000 \
--densify_grad_threshold 0.0001 \
--percent_dense 0.002 \
--opacity_reset_interval 9000"
    SCALE_DESC="28M"
fi


# Monitoring settings
MONITOR_OPTS="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Configure offload strategy
if [ "$offload_strategy" = "no_offload" ]; then
    offload_opts="--no_offload"
elif [ "$offload_strategy" = "naive_offload" ]; then
    offload_opts="--naive_offload"
elif [ "$offload_strategy" = "clm_offload" ]; then
    offload_opts="--clm_offload \
--prealloc_capacity 30_000_000"
fi

# Configure CUDA caching allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "========================================"
echo "Training Rubble 4K Scene"
echo "========================================"
echo "Strategy: $offload_strategy"
echo "Scale: $SCALE_DESC Gaussians"
echo "Batch size: $BSZ"
echo "Iterations: $ITERATIONS"
echo "========================================"
echo ""

# Run training
python train.py \
    -s ${dataset_folder} \
    ${DOWNSAMPLE_OPT} \
    --llffhold ${LLFFHOLD} \
    --log_folder ${log_folder} \
    --model_path ${model_path} \
    --iterations ${ITERATIONS} \
    --log_interval ${LOG_INTERVAL} \
    --bsz ${BSZ} \
    --test_iterations ${TEST_ITERATIONS} \
    --save_iterations ${SAVE_ITERATIONS} \
    --num_save_images_during_eval 5 \
    --save_tensors \
    ${DENSIFY_OPTS} \
    ${offload_opts} \
    ${MONITOR_OPTS} \
    --eval



if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "Results saved in: ${log_folder}"
echo "========================================"
