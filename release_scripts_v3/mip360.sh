#!/bin/bash

# mip360.sh - Training script for mip-NeRF 360 dataset
# This script trains all 7 scenes from the mip-NeRF 360 dataset with a specified offload strategy

# Check arguments
if [ $# -ne 2 ]; then
    echo "Error: Please specify exactly two arguments."
    echo "Usage: bash mip360.sh <dataset_folder> <offload_strategy>"
    echo ""
    echo "Arguments:"
    echo "  <dataset_folder>   : Path to the root folder containing mip360 scenes"
    echo "                       (should contain: counter, bicycle, stump, garden, room, bonsai, kitchen)"
    echo "  <offload_strategy> : One of: no_offload, naive_offload, clm_offload"
    echo ""
    echo "Example:"
    echo "  bash mip360.sh /path/to/mip360_dataset clm_offload"
    exit 1
fi

dataset_folder=$1
offload_strategy=$2

echo "Dataset folder: $dataset_folder"
echo "Offload strategy: $offload_strategy"

# Validate offload strategy
if [ "$offload_strategy" != "no_offload" ] && [ "$offload_strategy" != "naive_offload" ] && [ "$offload_strategy" != "clm_offload" ]; then
    echo "Error: Invalid offload strategy '$offload_strategy'"
    echo "Must be one of: no_offload, naive_offload, clm_offload"
    exit 1
fi

# Generate timestamp for experiment naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define scenes to train
SCENES=(counter bicycle stump garden room bonsai kitchen)
# SCENES=(bicycle garden)

# Training configurations
BSZ=4
ITERATIONS=30000
# ITERATIONS=3000
LOG_INTERVAL=250

# Test and save iterations
TEST_ITERATIONS="7000 30000"
# TEST_ITERATIONS="3000"
SAVE_ITERATIONS="7000 30000"
# SAVE_ITERATIONS=""

# Monitoring settings
MONITOR_OPTS="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Configure CUDA caching allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Loop through all scenes
for scene in ${SCENES[@]}; do
    echo ""
    echo "========================================"
    echo "Training scene: $scene"
    echo "========================================"
    
    # Experiment name
    expe_name="${scene}_${offload_strategy}"
    
    # Determine image folder based on scene
    # bicycle, stump, and garden use images_4; others use images_2
    if [ "$scene" = "bicycle" ] || [ "$scene" = "stump" ] || [ "$scene" = "garden" ]; then
        image_folder="images_4"
    else
        image_folder="images_2"
    fi
    
    # Set output paths
    log_folder="output/mip360/${TIMESTAMP}_${expe_name}"
    model_path=${log_folder}
    
    # Configure offload strategy
    if [ "$offload_strategy" = "no_offload" ]; then
        offload_opts="--no_offload"
    elif [ "$offload_strategy" = "naive_offload" ]; then
        offload_opts="--naive_offload"
    elif [ "$offload_strategy" = "clm_offload" ]; then
        offload_opts="--clm_offload \
--prealloc_capacity 7_000_000" # TODO: check whether 5M is enough for mip360? 5M is not enough for garden scene. 
    fi
    
    # Run training
    python train.py \
        -s ${dataset_folder}/${scene} \
        --images ${image_folder} \
        --llffhold 8 \
        --iterations ${ITERATIONS} \
        --log_interval ${LOG_INTERVAL} \
        --log_folder ${log_folder} \
        --model_path ${model_path} \
        --bsz ${BSZ} \
        --test_iterations ${TEST_ITERATIONS} \
        --save_iterations ${SAVE_ITERATIONS} \
        --num_save_images_during_eval 20 \
        ${offload_opts} \
        ${MONITOR_OPTS} \
        --eval



    if [ $? -ne 0 ]; then
        echo "Error: Training failed for scene $scene"
        exit 1
    fi
    
    echo "Completed training for scene: $scene"
done

echo ""
echo "========================================"
echo "All scenes completed successfully!"
echo "Results saved in: output/mip360/"
echo "========================================"

