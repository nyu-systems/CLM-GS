#!/bin/bash

# Run rendering (individual images)
python render_bigcity_images.py \
    -s /mnt/nvme0/dataset/matrixcity/big_city/aerial/pose/all_blocks \
    --bsz 4 \
    --load_pt_path /home/hexu/clm_release/Grendel-XS/output/bigcity/20251116_000539_bigcity_100m_clm_offload_102M/saved_tensors/iteration_499969 \
    --n_frames 3000 \
    --output_dir /home/hexu/clm_release/Grendel-XS/output/bigcity/20251116_000539_bigcity_100m_clm_offload_102M/render_images \
    --clm_offload \
    --gpu 1 \
    --prealloc_capacity 102_231_360 \
    --pointcloud_sample_rate 0.05 \
    --manual_height 15 \
    --save_video \
    --visualize_pointcloud

    # --manual_height 5.5 \ This is the max height of training view. 
    # Add --save_video flag if you also want to create a video

