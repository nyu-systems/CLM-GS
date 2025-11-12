#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Trajectory Rendering with Ground Truth Comparison for 3D Gaussian Splatting

This script performs rendering of trained 3D Gaussian Splatting models at original training camera
poses and compares them side-by-side with ground truth images. It calculates and displays PSNR
for each frame in the output video.

=== OVERVIEW ===
This script loads a trained 3D Gaussian Splatting model (saved as PLY point cloud files during
training) and renders at the original training camera poses. It's designed for quality evaluation
and validation of large-scale scenes that may not fit entirely in GPU memory by keeping Gaussian
parameters in CPU RAM and loading them to GPU on-demand.

=== KEY FEATURES ===
- Single GPU rendering (no distributed training/inference setup required)
- Three offloading strategies:
  * clm_offload: Chunk-based loading with optimized memory management (recommended for large scenes)
  * naive_offload: Simple CPU↔GPU transfers (baseline offload strategy)
  * no_offload: Keep all data on GPU (fastest but highest memory usage)
- On-demand GPU loading: Transfers data to GPU only during active rendering
- Original training poses: Uses actual training camera poses for validation
- Side-by-side comparison: Shows rendered image next to ground truth
- PSNR calculation: Displays PSNR value for each frame
- Video output: Generates MP4 videos at 30 FPS

=== MODEL LOADING ===
IMPORTANT: This script loads models from PLY files (saved during training with scene.save()),
NOT from checkpoint files (.pth). Checkpoint files contain optimizer states and other training
data that are not needed for rendering.

Model files are expected at: {model_path}/point_cloud/iteration_{N}/point_cloud.ply

During training, models are saved using:
- scene.save(iteration) → saves as PLY files (standard format, used by this script)
- scene.save_tensors(iteration) → saves as .pt files (optional fast loading format)

=== USAGE EXAMPLE ===
# Render with CLM offloading (recommended for large scenes):
python render_trajectory_with_gt.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration 30000 \\
    --clm_offload

# Render with no offloading (for scenes that fit in GPU memory):
# Use --iteration -1 to automatically load the latest saved iteration
python render_trajectory_with_gt.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration -1 \\
    --no_offload

# Limit to first 50 frames for quick testing:
python render_trajectory_with_gt.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration 30000 \\
    --max_frames 50 \\
    --clm_offload

=== TECHNICAL DETAILS ===
3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives. Each Gaussian has:
- Position: 3D coordinates (xyz) - 3 floats
- Rotation: Quaternion - 4 floats
- Scale: 3D anisotropic scaling - 3 floats
- Opacity: Transparency - 1 float
- Color: Spherical harmonics coefficients - (K coefficients × 3 RGB channels) floats

For large scenes (billions of Gaussians), storing all parameters on GPU exceeds available VRAM.
This script's offloading strategies enable rendering by:
1. Storing parameters in CPU RAM
2. Loading required data to GPU per frame (or per chunk)
3. Rendering the frame
4. Comparing with ground truth and calculating PSNR
5. Creating side-by-side visualization with PSNR overlay
6. Clearing GPU memory and moving to next frame

This allows quality evaluation of arbitrarily large scenes on consumer GPUs.
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional, List
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image

from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    init_args,
)

from scene import Scene, load_scene_info_for_rendering
from scene.cameras import Camera
from strategies.naive_offload import GaussianModelNaiveOffload, naive_offload_eval_one_cam
from strategies.clm_offload import GaussianModelCLMOffload, clm_offload_eval_one_cam
from strategies.no_offload import GaussianModelNoOffload, baseline_accumGrads_micro_step

# Trajectory generation not needed - using original training poses only

from utils.general_utils import safe_state, PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from utils.graphics_utils import getWorld2View2
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import loadCam
from utils.image_utils import psnr
import psutil
import time


def create_camera_from_cam_info(
    cam_info,
    uid: int = 0,
):
    """
    Create a Camera object from CameraInfo with ground truth image loaded from disk.
    
    Args:
        cam_info: CameraInfo object containing camera parameters and image_path
        uid: Camera unique identifier
    
    Returns:
        Camera object with ground truth image loaded
    """
    # Get args and log file
    args = get_args()
    log_file = get_log_file()
    
    # Get resolution
    resolution = utils.get_img_width(), utils.get_img_height()
    
    # Load image from disk (following loadCam pattern)
    image = Image.open(cam_info.image_path)
    resized_image_rgb = PILtoTorch(
        image, resolution, args, log_file, decompressed_image=None
    )
    
    # Take first 3 channels and make contiguous
    gt_image = resized_image_rgb[:3, ...].contiguous()
    loaded_mask = None
    
    # Free the memory
    image.close()
    image = None
    
    camera = Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=uid,
        offload=True,  # Keep camera on CPU
    )

    # if the system cpu memory is low, exit
    if psutil.virtual_memory().percent > 90: # 90%
        print("System CPU memory is low, exiting...")
        exit()
    return camera


def generate_trajectory_cameras(
    train_cameras: list,
):
    """
    Generate camera trajectory for rendering using original training poses.
    
    Args:
        train_cameras: List of CameraInfo objects from training set
    
    Returns:
        List of Camera objects with ground truth images for trajectory
    """
    # Check if we have training cameras
    if len(train_cameras) == 0:
        raise ValueError("No training cameras found in scene")
    
    print(f"Using original training poses: {len(train_cameras)} cameras")
    
    # Create Camera objects for each training camera (with GT images)
    trajectory_cameras = []
    for idx, cam_info in enumerate(train_cameras):
        camera = create_camera_from_cam_info(
            cam_info=cam_info,
            uid=idx,
        )
        trajectory_cameras.append(camera)
    
    print(f"Generated trajectory with {len(trajectory_cameras)} cameras with ground truth images")
    return trajectory_cameras


def render_trajectory_video(
    args,
    scene: Scene,
    gaussians,
    pipe_args,
    trajectory_cameras: List[Camera],
    output_path: str,
    background: torch.Tensor,
    max_frames: Optional[int] = None,
):
    """
    Render trajectory and save as video with side-by-side comparison and PSNR.
    
    Args:
        args: Command-line arguments
        scene: Scene object
        gaussians: Gaussian model
        pipe_args: Pipeline parameters
        trajectory_cameras: List of cameras for trajectory (with GT images)
        output_path: Path to save output video
        background: Background color tensor
        max_frames: Maximum number of frames to render (None = all)
    """
    # Limit number of frames if specified
    total_frames = len(trajectory_cameras)
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)
        trajectory_cameras = trajectory_cameras[:total_frames]
        print(f"Limiting to first {total_frames} frames")
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=30)
    
    print(f"Rendering {total_frames} frames...")
    
    # Render each frame
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        camera = trajectory_cameras[frame_idx]
        
        # Get ground truth image
        gt_image = camera.original_image_backup.clone().cuda()  # [3, H, W]
        
        # Prepare camera matrices on GPU
        camera.world_view_transform = camera.world_view_transform.cuda()
        camera.full_proj_transform = camera.full_proj_transform.cuda()
        camera.K = camera.create_k_on_gpu()
        camera.camtoworlds = torch.inverse(
            camera.world_view_transform.transpose(0, 1)
        ).unsqueeze(0)
        
        # Render frame based on offload strategy
        if args.naive_offload:
            rendered_image = naive_offload_eval_one_cam(
                gaussians=gaussians,
                scene=scene,
                camera=camera,
                background=background,
            )
        elif args.clm_offload:
            rendered_image = clm_offload_eval_one_cam(
                camera=camera,
                gaussians=gaussians,
                background=background,
                scene=scene,
            )
        elif args.no_offload:
            rendered_image, _, _, _ = baseline_accumGrads_micro_step(
                means3D=gaussians.get_xyz,
                opacities=gaussians.get_opacity,
                scales=gaussians.get_scaling,
                rotations=gaussians.get_rotation,
                shs=gaussians.get_features,
                sh_degree=gaussians.active_sh_degree,
                camera=camera,
                background=background,
                mode="eval",
            )
        else:
            raise ValueError("Invalid offload configuration")
        
        # Process rendered output
        rendered_colors = torch.clamp(rendered_image, 0.0, 1.0)  # Normalize to [0, 1]
        gt_colors = torch.clamp(gt_image / 255.0, 0.0, 1.0)  # Normalize to [0, 1]
        
        # Calculate PSNR (images should be in [C, H, W] format with batch dimension)
        rendered_batch = rendered_colors.unsqueeze(0)  # [1, C, H, W] or [1, H, W, C]
        gt_batch = gt_colors.unsqueeze(0)  # [1, C, H, W]
        
        # Ensure both are in [1, C, H, W] format
        if rendered_batch.shape[1] != 3 and rendered_batch.shape[-1] == 3:
            rendered_batch = rendered_batch.permute(0, 3, 1, 2)  # [1, H, W, 3] -> [1, 3, H, W]
        
        psnr_value = psnr(rendered_batch, gt_batch).mean().item()
        
        # Convert to [H, W, 3] format for display
        if rendered_colors.shape[0] == 3:
            rendered_colors = rendered_colors.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        if gt_colors.shape[0] == 3:
            gt_colors = gt_colors.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # Convert to numpy uint8
        rendered_np = (rendered_colors * 255).cpu().to(torch.uint8).numpy()
        gt_np = (gt_colors * 255).cpu().to(torch.uint8).numpy()
        
        # Create side-by-side comparison
        h, w = rendered_np.shape[:2]
        side_by_side = np.concatenate([rendered_np, gt_np], axis=1)  # [H, 2*W, 3]
        
        # Add PSNR text overlay using OpenCV
        side_by_side = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"PSNR: {psnr_value:.2f} dB"
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Position text at top center
        text_x = (side_by_side.shape[1] - text_size[0]) // 2
        text_y = 30
        
        # Add background rectangle for text
        padding = 5
        cv2.rectangle(side_by_side, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(side_by_side, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Add labels for rendered and GT
        label_font_scale = 0.7
        label_thickness = 2
        cv2.putText(side_by_side, "Rendered", (10, h - 10),
                   font, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
        cv2.putText(side_by_side, "Ground Truth", (w + 10, h - 10),
                   font, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
        
        # Convert back to RGB for video writer
        side_by_side = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)
        
        # Write frame to video
        writer.append_data(side_by_side)
        
        # Clean up GPU memory
        del rendered_image, rendered_colors, gt_colors, rendered_np, gt_np, side_by_side
        camera.world_view_transform = camera.world_view_transform.cpu()
        camera.full_proj_transform = camera.full_proj_transform.cpu()
        torch.cuda.empty_cache()
    
    writer.close()
    print(f"Saved video: {output_path}")


def main():
    """Main function for trajectory rendering with ground truth comparison."""
    # Parse command-line arguments
    parser = ArgumentParser(description="Trajectory rendering script with GT comparison")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    
    # Add trajectory-specific arguments
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to render (None = all)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/render_traj_gt)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (default: -1 = load latest saved iteration)")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize arguments
    init_args(args)
    args = utils.get_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "render_traj_gt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = open(os.path.join(args.output_dir, "render_traj_gt.log"), "w")
    utils.set_log_file(log_file)
    
    utils.print_rank_0("=" * 80)
    utils.print_rank_0("Trajectory Rendering with Ground Truth Comparison")
    utils.print_rank_0("=" * 80)
    utils.print_rank_0(f"Model path: {args.model_path}")
    utils.print_rank_0(f"Load iteration: {args.iteration} (-1 = latest)")
    utils.print_rank_0(f"Trajectory type: original (training poses)")
    utils.print_rank_0(f"Output directory: {args.output_dir}")
    utils.print_rank_0("=" * 80)
    
    # Initialize Gaussian model based on offload strategy
    dataset_args = lp.extract(args)
    
    offload_strategy = None
    if args.naive_offload:
        gaussians = GaussianModelNaiveOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNaiveOffload")
        offload_strategy = "naive_offload"
    elif args.clm_offload:
        gaussians = GaussianModelCLMOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelCLMOffload")
        offload_strategy = "clm_offload"
    elif args.no_offload:
        gaussians = GaussianModelNoOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNoOffload (no offload, GPU-only)")
        offload_strategy = "no_offload"
    else:
        raise ValueError(
            f"Invalid offload configuration: naive_offload={args.naive_offload}, "
            f"clm_offload={args.clm_offload}, no_offload={args.no_offload}"
        )
    
    # Load scene info and trained model
    # Note: We load camera metadata WITHOUT decoding images to disk (rendering-only workflow)
    # The PLY files are saved at: model_path/point_cloud/iteration_{iteration}/point_cloud.ply
    with torch.no_grad():
        utils.print_rank_0(f"\nLoading scene info and model from iteration {args.iteration}...")
        
        # Load scene information (camera poses, intrinsics) without image decoding
        scene_info, cameras_extent = load_scene_info_for_rendering(args)
        utils.print_rank_0("Loaded scene metadata (camera poses and intrinsics)")
        
        # Determine which iteration to load
        if args.iteration == -1:
            loaded_iter = searchForMaxIteration(
                os.path.join(args.model_path, "point_cloud")
            )
        else:
            loaded_iter = args.iteration
        
        utils.print_rank_0(f"Loading Gaussians from iteration {loaded_iter}...")
        
        # Manually load the trained Gaussian model from PLY file
        ply_path = os.path.join(
            args.model_path, "point_cloud", f"iteration_{loaded_iter}"
        )
        gaussians.load_ply(ply_path)
        
        utils.print_rank_0(f"Loaded {len(gaussians._xyz)} Gaussians from iteration {loaded_iter}")
        
        # Set active SH degree to maximum for best quality rendering
        # During training, SH degree is progressively increased
        # For rendering, we use the maximum degree for best visual quality
        gaussians.active_sh_degree = gaussians.max_sh_degree
    
    # know the image width and height
    image_width = scene_info.train_cameras[0].width
    image_height = scene_info.train_cameras[0].height

    # utils.get_img_width()
    utils.set_img_size(image_height, image_width)

    # Setup background
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Generate trajectory cameras from training cameras with ground truth images
    utils.print_rank_0("Loading training cameras with ground truth images...")
    trajectory_cameras = generate_trajectory_cameras(
        train_cameras=scene_info.train_cameras,
    )
    
    # Render trajectory video with GT comparison
    pipe_args = pp.extract(args)
    video_path = os.path.join(args.output_dir, f"trajectory_original_with_gt_{offload_strategy}.mp4")
    
    render_trajectory_video(
        args=args,
        scene=None,  # No scene object needed for rendering
        gaussians=gaussians,
        pipe_args=pipe_args,
        trajectory_cameras=trajectory_cameras,
        output_path=video_path,
        background=background,
        max_frames=args.max_frames,
    )
    
    # Cleanup
    log_file.write(f"Rendering completed. Video saved to: {video_path}\n")
    log_file.close()
    utils.print_rank_0("\nRendering complete!")
    utils.print_rank_0(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
