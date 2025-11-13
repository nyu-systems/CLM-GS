#!/bin/bash

# bigcity.sh - Training script for MatrixCity BigCity dataset
# This demonstrates the extreme-scale capability upper bound of the Grendel-XS system
# Note: This dataset is synthetic and designed to showcase maximum scalability

# Check arguments
if [ $# -ne 1 ]; then
    echo "Error: Please specify exactly one argument."
    echo "Usage: bash bigcity.sh <dataset_folder>"
    echo ""
    echo "Arguments:"
    echo "  <dataset_folder> : Path to the MatrixCity BigCity dataset folder"
    echo "                     (e.g., /path/to/matrixcity/big_city/aerial/pose/all_blocks)"
    echo ""
    echo "Example:"
    echo "  bash bigcity.sh /path/to/matrixcity/big_city/aerial/pose/all_blocks"
    echo ""
    echo "Note: This script uses only clm_offload strategy due to extreme scale (102M+ Gaussians)"
    exit 1
fi

dataset_folder=$1

echo "Dataset folder: $dataset_folder"
echo "Offload strategy: clm_offload (required for extreme scale)"

# Generate timestamp for experiment naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Experiment name
expe_name="bigcity_102m_clm_offload"

# Set output paths
log_folder="output/bigcity/${TIMESTAMP}_${expe_name}"
model_path=${log_folder}

echo "Output folder: $log_folder"

# Training configurations
BSZ=64
ITERATIONS=500000
# ITERATIONS=100
LOG_INTERVAL=250

# Test and save iterations
TEST_ITERATIONS="100000 200000 300000 400000 500000"
SAVE_ITERATIONS="200000 500000"
# SAVE_ITERATIONS="100"

# Densification parameters - DISABLED for extreme scale
# We use pre-allocated capacity instead of auto-densification
DENSIFY_OPTS="--disable_auto_densification"

# Offload configuration - CLM offload with pre-allocated capacity
# Pre-allocate for 102M+ Gaussians
OFFLOAD_OPTS="--clm_offload \
--prealloc_capacity 102_231_360 \
--sparse_adam"

# Monitoring settings
MONITOR_OPTS="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Configure CUDA caching allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Enable detailed error tracking (uncomment if debugging)
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1

echo ""
echo "========================================"
echo "Training MatrixCity BigCity (Extreme Scale)"
echo "========================================"
echo "Strategy: clm_offload"
echo "Batch size: $BSZ"
echo "Iterations: $ITERATIONS"
echo "Pre-allocated capacity: 102,231,360 Gaussians"
echo "========================================"
echo ""
echo "WARNING: This is an extreme-scale experiment that will take"
echo "5-10 hours and requires significant computational resources."
echo "Press Ctrl+C within 5 seconds to cancel..."
echo ""
sleep 5

# Run training
python train.py \
    -s ${dataset_folder} \
    --log_folder ${log_folder} \
    --model_path ${model_path} \
    --iterations ${ITERATIONS} \
    --log_interval ${LOG_INTERVAL} \
    --bsz ${BSZ} \
    --test_iterations ${TEST_ITERATIONS} \
    --save_iterations ${SAVE_ITERATIONS} \
    ${DENSIFY_OPTS} \
    ${OFFLOAD_OPTS} \
    ${MONITOR_OPTS} \
    --eval \
    --save_tensors \
    --num_save_images_during_eval 10 \
    --max_num_images_to_evaluate 100

# save_tensors is very important for matrixcity. 

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "Results saved in: ${log_folder}"
echo "========================================"

