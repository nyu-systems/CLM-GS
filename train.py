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

import os
import sys
import json
import gc
import psutil
# import faulthandler

# faulthandler_log = open("fault.log", "w")
# faulthandler.enable(file=faulthandler_log, all_threads=True)

import torch
import torch.multiprocessing
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)

from scene import Scene, OffloadSceneDataset
from strategies.naive_offload import GaussianModelNaiveOffload, naive_offload_train_one_batch, naive_offload_eval_one_cam
from strategies.clm_offload import GaussianModelCLMOffload, clm_offload_train_one_batch, clm_offload_eval_one_cam
from strategies.no_offload import GaussianModelNoOffload, baseline_accumGrads_impl, baseline_accumGrads_micro_step

from utils.general_utils import safe_state, prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

from densification import gsplat_densification


def training(dataset_args, opt_args, pipe_args, args, log_file):
    """
    Main training loop for Gaussian Splatting with parameter offloading.
    
    This function orchestrates the entire training process:
    1. Initialize scene, gaussians, and data loaders
    2. Main iteration loop: load data → forward/backward → optimize → densify
    3. Periodic evaluation, checkpointing, and logging
    4. Cleanup and final statistics reporting
    
    The training supports two main offloading strategies:
    - braindeath_offload: Simple baseline with bulk parameter transfers
    - pipelined_offload: Sophisticated retention-based approach with overlapped comm/compute
    """
    
    # ============================================================================
    # STAGE 1: INITIALIZATION
    # ============================================================================
    
    # FIXME: add support for load_from_cpuram_on_demand
    assert args.dataset_cache_and_stream_mode in ["load_from_disk_on_demand"], f"Only load_from_disk_on_demand is supported for now, but got {args.dataset_cache_and_stream_mode}"

    # ------------------------------------------------------------------------
    # 1.1: Setup auxiliary tools and GPU configuration
    # ------------------------------------------------------------------------
    gc.disable()  # Disable Python GC for better performance control

    torch.cuda.set_device(args.gpu)
    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1
    
    # Configure multiprocessing sharing strategy if needed
    if args.sharing_strategy != "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)

    # ------------------------------------------------------------------------
    # 1.2: Initialize scene and gaussian model
    # ------------------------------------------------------------------------
    # Select the appropriate Gaussian model based on offload strategy
    if args.naive_offload:
        gaussians = GaussianModelNaiveOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNaiveOffload")
        log_file.write("Using GaussianModelNaiveOffload\n")
    elif args.clm_offload:
        gaussians = GaussianModelCLMOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelCLMOffload")
        log_file.write("Using GaussianModelCLMOffload\n")
    elif args.no_offload:
        gaussians = GaussianModelNoOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNoOffload (no offload, GPU-only)")
        log_file.write("Using GaussianModelNoOffload (no offload, GPU-only)\n")
    else:
        raise ValueError(f"Invalid offload configuration: naive_offload={args.naive_offload}, clm_offload={args.clm_offload}, no_offload={args.no_offload}")

    with torch.no_grad():
        scene = Scene(args, gaussians)
        gaussians.training_setup(opt_args)

        # Restore from checkpoint if specified
        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0(
                "Restored from checkpoint: {}".format(args.start_checkpoint)
            )
            log_file.write(
                "Restored from checkpoint: {}\n".format(args.start_checkpoint)
            )

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # ------------------------------------------------------------------------
    # 1.3: Initialize data loader
    # ------------------------------------------------------------------------
    train_dataset = OffloadSceneDataset(scene.getTrainCamerasInfo())
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        num_workers=1,          # Single worker for sequential loading
        shuffle=True,            # Randomize camera order
        drop_last=True,          # Drop incomplete batches
        persistent_workers=True, # Keep workers alive between epochs
        pin_memory=True,         # Enable faster GPU transfers
        collate_fn=(lambda batch: batch)
    )
    dataloader_iter = iter(dataloader)

    # ------------------------------------------------------------------------
    # 1.4: Initialize background and CUDA streams
    # ------------------------------------------------------------------------
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
    # Dedicated stream for CPU↔GPU communication (overlapped with compute)
    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

    # ------------------------------------------------------------------------
    # 1.5: Initialize training loop state
    # ------------------------------------------------------------------------
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress"
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    # Random number generator for camera ordering in retention-based offloading
    perm_generator = torch.Generator(device="cuda")
    perm_generator.manual_seed(1)

    # Training state variables
    ema_loss_for_log = 0
    means3D_all = None          # Handle to means3D on GPU (for densification)
    send2gpu_filter = None      # Handle to send2gpu_filter on GPU
    send2gpu_filter_cpu = None  # Handle to send2gpu_filter on CPU
    # ============================================================================
    # STAGE 2: MAIN TRAINING LOOP
    # ============================================================================
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # # rewrite the checking iterations
        # ------------------------------------------------------------------------
        # 2.1: Iteration setup and profiling
        # ------------------------------------------------------------------------
        # Optional: trace CUDA memory usage for debugging
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (iteration % args.densification_interval) == 0:
                torch.cuda.memory._record_memory_history()
                log_file.write(
                    "[ITER {}] Tracing cuda memory usage.\n".format(iteration)
                )
        
        # Update progress bar and iteration state
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)  # Learning rate scheduling
        num_trained_batches += 1
        
        # Optional: reset memory tracking for per-iteration profiling
        if args.reset_each_iter:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
        
        # Start timing this iteration
        timers.clear()
        timers.start("[iteration end2end]")
        
        # Optional: NSight Systems profiling
        if args.nsys_profile:
            if iteration == args.nsys_profile_start_iter:
                torch.cuda.cudart().cudaProfilerStart()
            if iteration == args.nsys_profile_end_iter or iteration == opt_args.iterations:
                torch.cuda.cudart().cudaProfilerStop()
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
        
        # Gradually increase spherical harmonics degree (every 1000 iterations)
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # ------------------------------------------------------------------------
        # 2.2: Load training data (camera images)
        # ------------------------------------------------------------------------
        timers.start("dataloader: load the next image from disk and decode")
        try:
            batched_cameras = next(dataloader_iter)
        except StopIteration:
            # Reached end of dataset, restart from beginning
            dataloader_iter = iter(dataloader)
            batched_cameras = next(dataloader_iter)
        timers.stop("dataloader: load the next image from disk and decode")
        
        # Assign unique IDs within the batch (for tracking)
        # TODO: camera.uid should be the id within a batch. Currently we use a postfix as workaround.
        for uid, c in enumerate(batched_cameras):
            c.uid = uid
        
        # ------------------------------------------------------------------------
        # 2.3: Transfer camera matrices to GPU
        # ------------------------------------------------------------------------
        timers.start("send cam matrices to gpu")
        # Transfer world-view and projection transforms
        for camera in batched_cameras:
            camera.world_view_transform = camera.world_view_transform.cuda()
            camera.full_proj_transform = camera.full_proj_transform.cuda()

        # Create camera intrinsics (K matrix) and compute camera-to-world transforms
        batched_world_view_transform = []
        for camera in batched_cameras:
            camera.K = camera.create_k_on_gpu()
            batched_world_view_transform.append(camera.world_view_transform.transpose(0, 1))
        
        # Batch process: compute inverse transforms for all cameras
        batched_world_view_transform = torch.stack(batched_world_view_transform)
        batched_world_view_transform_inverse = torch.inverse(batched_world_view_transform)
        batched_world_view_transform_inverse = torch.unbind(batched_world_view_transform_inverse, dim=0)
        
        # Store camera-to-world transforms (for view direction computation)
        for camera, wvt in zip(batched_cameras, batched_world_view_transform_inverse):
            camera.camtoworlds = wvt.unsqueeze(0)
        # TODO: maybe we can save them on GPU during initialization. After all, they do not take up lots of memory.
        timers.stop("send cam matrices to gpu")
            
        # ------------------------------------------------------------------------
        # 2.4: Load ground-truth images to GPU
        # ------------------------------------------------------------------------
        with torch.no_grad():
            timers.start("load_cameras")
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
            timers.stop("load_cameras")
        # ------------------------------------------------------------------------
        # 2.5: Forward/Backward Pass - Choose offloading strategy
        # ------------------------------------------------------------------------
        if args.naive_offload:
            # BASELINE: Simple bulk parameter transfer strategy
            # Load all params → process all cameras → offload all gradients
            N = gaussians._xyz.shape[0]

            losses, visibility = naive_offload_train_one_batch(
                gaussians,
                scene,
                batched_cameras,
                background,
                sparse_adam=args.sparse_adam,
            )
            batched_screenspace_pkg = {}

            # Aggregate and log losses
            timers.start("sync_loss_and_log")
            batched_losses = torch.stack(losses)
            batched_loss_cpu = batched_losses.cpu().numpy()
            
            # Exponential moving average for smoother loss tracking
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            
            # Update dataset statistics (for adaptive sampling)
            train_dataset.update_losses(batched_loss_cpu)
            
            # Log iteration results
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}), loss: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)

        elif args.clm_offload:
            # OPTIMIZED: Retention-based parameter offloading with overlapped comm/compute
            # Selective loading → retention across cameras → concurrent CPU Adam
            assert args.bsz > 1, "Pipelined offload requires batch size > 1"

            N = gaussians._xyz.shape[0]

            losses, ordered_cams, sparsity = clm_offload_train_one_batch(
                gaussians,
                scene,
                batched_cameras,
                gaussians.parameters_grad_buffer,
                background,
                pipe_args,
                comm_stream,
                perm_generator,
            )

            batched_screenspace_pkg = {}
            # Reorder cameras to match the optimized processing order
            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            
            # Aggregate and log losses
            timers.start("sync_loss_and_log")
            batched_losses = torch.stack(losses)
            batched_loss_cpu = batched_losses.cpu().numpy()
            
            # Exponential moving average for smoother loss tracking
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            
            # Update dataset statistics (for adaptive sampling)
            train_dataset.update_losses(batched_loss_cpu)
            
            # Log iteration results (including sparsity metric)
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                sparsity,  # Measures parameter reuse efficiency
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)

        elif args.no_offload:# This is GPU baseline. 
            losses, visibility = baseline_accumGrads_impl(
                gaussians,
                scene,
                batched_cameras,
                background,
                sparse_adam=args.sparse_adam
            )
            batched_screenspace_pkg = {}
            
            batched_loss = torch.stack(losses)
            batched_loss_cpu = batched_loss.cpu().numpy()
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            # Update Epoch Statistics
            train_dataset.update_losses(batched_loss_cpu)
            # Logging
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)
        else:
            raise ValueError("Invalid configuration")


        with torch.no_grad():
            # ------------------------------------------------------------------------
            # 2.6: Periodic evaluation on validation set
            # ------------------------------------------------------------------------
            end2end_timers.stop()
            training_report(
                iteration,
                l1_loss,
                args.test_iterations,
                scene,
                pipe_args,
                background,
            )
            end2end_timers.start()

            # ------------------------------------------------------------------------
            # 2.7: Adaptive densification (add/split/prune gaussians)
            # ------------------------------------------------------------------------
            gsplat_densification(
                iteration, scene, gaussians, batched_screenspace_pkg
            )
            
            # Perform actual densification at specified intervals
            if not args.disable_auto_densification and iteration <= args.densify_until_iter and iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                iteration, args.bsz, args.densification_interval, 0
            ):
                # Invalidate cached data (gaussians may be added/removed)
                means3D_all = None
                send2gpu_filter = None
                send2gpu_filter_cpu = None
            
            # ------------------------------------------------------------------------
            # 2.8: Free temporary activation states
            # ------------------------------------------------------------------------
            batched_screenspace_pkg = None
            batched_image = None
            batched_compute_locally = None

            # ------------------------------------------------------------------------
            # 2.9: Save trained gaussians (if at save iteration)
            # ------------------------------------------------------------------------
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                utils.print_rank_0("\n[ITER {}] Saving End2end".format(iteration))
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)

                if not args.do_not_save:
                    utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                    log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                    
                    if args.save_tensors:
                        # Save as PyTorch tensors (.pt format) for faster loading
                        utils.print_rank_0("NOTE: Saving model as .pt files instead of .ply file.")
                        scene.save_tensors(iteration)
                    else:
                        # Save as PLY point cloud (standard format)
                        scene.save(iteration)

                end2end_timers.start()

            # ------------------------------------------------------------------------
            # 2.10: Save training checkpoint (for resuming)
            # ------------------------------------------------------------------------
            if any(
                [
                    iteration <= checkpoint_iteration < iteration + args.bsz
                    for checkpoint_iteration in args.checkpoint_iterations
                ]
            ):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                os.makedirs(save_folder, exist_ok=True)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),  # Model state + iteration number
                    save_folder + "/chkpnt.pth"
                )
                end2end_timers.start()

            # ------------------------------------------------------------------------
            # 2.11: Optimizer step (for non-overlapped strategies only)
            # ------------------------------------------------------------------------
            # Note: For pipelined_offload and braindeath_offload, optimizer step
            # is performed inside their respective implementations
            if iteration < opt_args.iterations and not args.clm_offload and not args.naive_offload:
                timers.start("optimizer_step")

                # Scale gradients by batch size (unless using gradient accumulation mode)
                torch.cuda.nvtx.range_push("scale grad")
                if args.lr_scale_mode != "accumu":
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz
                torch.cuda.nvtx.range_pop()

                # Apply optimizer update
                if not args.stop_update_param:
                    torch.cuda.nvtx.range_push("optimizer step")
                    if args.sparse_adam:
                        # Sparse Adam: only update parameters that received gradients
                        if args.naive_offload:
                            sparse_indices = torch.nonzero(visibility.squeeze()).flatten().to(torch.int32)
                            sparse_indices = sparse_indices.to("cpu")
                            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
                        elif args.clm_offload:
                            gaussians.optimizer.step(visibility=visibility)
                        else:
                            raise ValueError("Invalid offload value")
                        del visibility
                    else:
                        # Dense Adam: update all parameters
                        gaussians.optimizer.step()
                    torch.cuda.nvtx.range_pop()
                
                # Clear gradients for next iteration
                torch.cuda.nvtx.range_push("zero_grad")
                gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.nvtx.range_pop()
                timers.stop("optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")
                
                # Clear gradient buffer for offloading strategies
                if args.clm_offload:
                    timers.start("zero out grads")
                    gaussians.parameters_grad_buffer[:N, :].zero_()
                    timers.stop("zero out grads")

        # ------------------------------------------------------------------------
        # 2.12: Iteration cleanup
        # ------------------------------------------------------------------------
        torch.cuda.synchronize()  # Ensure all GPU operations are complete
        
        # Release camera image memory
        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None
        
        # End profiling range if active
        if args.nsys_profile:
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_pop()
        
        # Print timing statistics
        if utils.check_enable_python_timer():
            timers.stop("[iteration end2end]")
            timers.printTimers(iteration, mode="sum")
        
        # Dump CUDA memory trace if enabled
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (iteration % args.densification_interval) == 0:
                dump_name = args.model_path + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            
        utils.memory_report("at the end of the iteration")
        log_file.flush()

    # ============================================================================
    # STAGE 3: POST-TRAINING CLEANUP AND REPORTING
    # ============================================================================
    
    # Clean up CUDA resources
    del comm_stream
    
    # Print final timing statistics
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    
    # Log peak memory usage
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    
    # Close progress bar and clean up scene
    progress_bar.close()
    scene.clean_up()
    
    # Stop profiler if active
    if args.nsys_profile:
        torch.cuda.cudart().cudaProfilerStop()
    
    # Log CPU Adam trailing overhead statistics (if measured)
    if args.log_cpu_adam_trailing_overhead:
        # Calculate average overhead (excluding first 3 warmup iterations)
        average_cpu_adam_trailing_from_default_stream = (
            args.cpu_adam_trailing_overhead["from_default_stream"] / (args.cpu_adam_trailing_overhead["step"] - 3)
        )
        average_cpu_adam_trailing_from_comm_stream = (
            args.cpu_adam_trailing_overhead["from_comm_stream"] / (args.cpu_adam_trailing_overhead["step"] - 3)
        )

        # Trailing overhead: time spent waiting for CPU Adam to finish after GPU is done
        log_file.write(
            "CPU Adam trailing [from default stream]: {} ms.\n".format(average_cpu_adam_trailing_from_default_stream)
        )
        log_file.write(
            "CPU Adam trailing [from comm stream]: {} ms.\n".format(average_cpu_adam_trailing_from_comm_stream)
        )



def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background
):
    args = utils.get_args()
    log_file = utils.get_log_file()
    # Report test and samples of training set
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(
        iteration, utils.get_args().bsz, testing_iterations[0], 0
    ):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))

        validation_configs = (
            {
                "name": "test", 
                "cameras": scene.getTestCameras(), 
                "cameras_info": scene.getTestCamerasInfo(),
                "num_cameras": len(scene.getTestCameras() if scene.getTestCameras() is not None else scene.getTestCamerasInfo()),
            },
            {
                "name": "train",
                "cameras": scene.getTrainCameras(),
                "cameras_info": scene.getTrainCamerasInfo(),
                "num_cameras": max(len(scene.getTrainCameras() if scene.getTrainCameras() is not None else scene.getTrainCamerasInfo()) // args.llffhold, 1),
            },
        )

        # init workload division strategy
        for config in validation_configs:
            # Dataset is offloaded to disk
            l1_test = torch.scalar_tensor(0.0, device="cuda")
            psnr_test = torch.scalar_tensor(0.0, device="cuda")

            # TODO: if not divisible by world size
            num_cameras = min(config["num_cameras"], args.max_num_images_to_evaluate)
            eval_dataset = OffloadSceneDataset(config["cameras_info"])
            # Init dataloader: num_workers = 0
            dataloader = DataLoader(
                eval_dataset,
                batch_size=1,
                # shuffle=True,
                num_workers=1,
                pin_memory=True,
                collate_fn=(lambda batch: batch)
            )
            dataloader_iter = iter(dataloader)
            
            for idx in range(1, num_cameras + 1, 1):
                num_camera_to_load = min(1, num_cameras - idx + 1)
                #FIXME: may have problems when bsz > 1
                try:
                    batched_cameras = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batched_cameras = next(dataloader_iter)
                # batched_cameras = eval_dataset.get_batched_cameras(
                #     num_camera_to_load
                # )
                # Load ground-truth images to GPU
                for camera in batched_cameras:
                    camera.original_image = camera.original_image_backup.cuda()
                
                batched_image = []
                for cam_id, camera in enumerate(batched_cameras):
                    #FIXME: quick workaround for verifying the correctness
                    camera.world_view_transform = camera.world_view_transform.cuda()
                    camera.full_proj_transform = camera.full_proj_transform.cuda()
                    camera.K = camera.create_k_on_gpu()
                    camera.camtoworlds = torch.inverse(camera.world_view_transform.transpose(0, 1)).unsqueeze(0)

                    if args.naive_offload:
                        rendered_image = naive_offload_eval_one_cam(
                            camera=camera,
                            gaussians=scene.gaussians,
                            background=background,
                            scene=scene
                        )
                        batched_image.append(rendered_image)
                    
                    elif args.clm_offload:
                        rendered_image = clm_offload_eval_one_cam(
                            camera=camera,
                            gaussians=scene.gaussians,
                            background=background,
                            scene=scene
                        )
                        batched_image.append(rendered_image)
                    
                    elif args.no_offload:
                        rendered_image, _, _, _ = baseline_accumGrads_micro_step(
                            means3D=scene.gaussians.get_xyz,
                            opacities=scene.gaussians.get_opacity,
                            scales=scene.gaussians.get_scaling,
                            rotations=scene.gaussians.get_rotation,
                            shs=scene.gaussians.get_features,
                            sh_degree=scene.gaussians.active_sh_degree,
                            camera=camera,
                            background=background,
                            mode="eval"
                        )
                        batched_image.append(rendered_image)
                    else:
                        raise ValueError("Invalid offload value")


                for camera_id, (image, gt_camera) in enumerate(
                    zip(batched_image, batched_cameras)
                ):
                    if (
                        image is None or len(image.shape) == 0
                    ):  # The image is not rendered locally.
                        image = torch.zeros(
                            gt_camera.original_image.shape,
                            device="cuda",
                            dtype=torch.float32,
                        )

                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(
                        gt_camera.original_image / 255.0, 0.0, 1.0
                    )

                    if idx + camera_id < num_cameras + 1:
                        l1_test += l1_loss(image, gt_image).mean().double().item()
                        psnr_test += psnr(image, gt_image).mean().double().item()
                    
                    # Save rendered and ground-truth images
                    if idx < args.num_save_images_during_eval:
                        # Create output directories for train/test
                        save_dir = os.path.join(args.model_path, config["name"])
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Get the original image name (without extension)
                        img_name = gt_camera.image_name.replace("/", "_")
                        
                        # Create filenames: {iteration}_{original_name}_render.png and {iteration}_{original_name}_gt.png
                        render_filename = f"{iteration:06d}_{img_name}_render.png"
                        gt_filename = f"{iteration:06d}_{img_name}_gt.png"
                        
                        # Save rendered image
                        torchvision.utils.save_image(
                            image,
                            os.path.join(save_dir, render_filename)
                        )
                        
                        # Save ground-truth image
                        torchvision.utils.save_image(
                            gt_image,
                            os.path.join(save_dir, gt_filename)
                        )
                    
                    gt_camera.original_image = None
            
            psnr_test /= num_cameras
            l1_test /= num_cameras
            utils.print_rank_0(
                "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config["name"], l1_test, psnr_test
                )
            )
            log_file.write(
                "[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(
                    iteration, config["name"], l1_test, psnr_test
                )
            )
                

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    args = utils.get_args()

    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    with open(args.log_folder + "/args.json", "w") as f:
        json.dump(vars(args), f)

    # create cuda trace dump folder
    if args.trace_cuda_mem:
        os.makedirs(os.path.join(args.model_path, "trace_dump"))
    

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(
        args.log_folder + "/python.log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    p = psutil.Process()
    log_file.write(f"Initial pinned memory: {p.memory_info().shared / 1024 / 1024 / 1024} GB\n")

    training(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )

    # All done
    utils.print_rank_0("\nTraining complete.")

# clm_offload, naive_offload, no_offload
