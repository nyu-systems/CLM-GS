import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
    gsplat_distributed_preprocess3dgs_and_all2all_final,
    gsplat_render_final,
)
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from scene import Scene, GaussianModel, SceneDataset, TorchSceneDataset, custom_collate_fn
from gaussian_renderer.workload_division import (
    start_strategy_final,
    finish_strategy_final,
    DivisionStrategyHistoryFinal,
)
from gaussian_renderer.loss_distribution import (
    load_camera_from_cpu_to_all_gpu,
    load_camera_from_cpu_to_all_gpu_for_eval,
    batched_loss_computation,
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification, gsplat_densification
from diff_gaussian_rasterization import (
    send2cpu,
    send2cpu_cat_buffer,
    send2cpu_cat_buffer_osr_shs,
)
import torch.multiprocessing

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1
    if args.sharing_strategy is not "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)

    # Init parameterized scene
    gaussians = GaussianModel(sh_degree=dataset_args.sh_degree, offload=args.offload, mxw_debug=args.mxw_debug)

    with torch.no_grad():
        if args.torch_dataloader:
            scene = Scene(args, gaussians, shuffle=False) #HACK: temporarily disable shuffling for reproductability
        else:
            scene = Scene(args, gaussians)
        gaussians.training_setup(opt_args)

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

    # Init dataset
    if args.torch_dataloader:
        train_dataset = TorchSceneDataset(scene.getTrainCameras(), scene.getTrainCamerasInfo())
        if args.num_workers == 0:
            dataloader = DataLoader(
                train_dataset,
                batch_size=args.bsz,
                # shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
        elif args.num_workers > 0:
            dataloader = DataLoader(
                train_dataset,
                batch_size=args.bsz,
                num_workers=args.num_workers,
                # shuffle=True,
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
        else:
            assert False, "`num_workers` should be a positive number"
        dataloader_iter = iter(dataloader)
    else:
        train_dataset = SceneDataset(scene.getTrainCameras(), scene.getTrainCamerasInfo())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = train_dataset.camera_size
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.

    # Init distribution strategy history
    strategy_history = DivisionStrategyHistoryFinal(
        train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    # Init background
    background = None
    if args.backend == "gsplat":
        bg_color = [1, 1, 1] if dataset_args.white_background else None
    else:
        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
    # Preallocate memory for grad
    #TODO: replace hard-coded dim
    if args.offload and args.mxw_debug == 'fused':
        means3D_grad_buffer = torch.empty((args.prealloc_capacity, 3), dtype=torch.float32, pin_memory=True)
        opacities_grad_buffer = torch.empty((args.prealloc_capacity, 1), dtype=torch.float32, pin_memory=True)
        scales_grad_buffer = torch.empty((args.prealloc_capacity, 3), dtype=torch.float32, pin_memory=True)
        rotations_grad_buffer = torch.empty((args.prealloc_capacity, 4), dtype=torch.float32, pin_memory=True)
        features_dc_grad_buffer = torch.empty((args.prealloc_capacity, 1, 3), dtype=torch.float32, pin_memory=True)
        features_rest_grad_buffer = torch.empty((args.prealloc_capacity, 15, 3), dtype=torch.float32, pin_memory=True)
    elif args.offload and args.mxw_debug == 'cat':
        parameters_grad_buffer = torch.zeros((args.prealloc_capacity, 59), dtype=torch.float32, pin_memory=True)

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    ema_loss_for_log = 0
    means3D_all = None # A handle to means3D_all on gpu
    send2gpu_filter = None # A handle to send2gpu_filter on gpu
    send2gpu_filter_cpu = None # A handle to send2gpu_filter on cpu
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        if args.offload and args.mxw_debug == 'fused':
            assert gaussians._xyz.is_pinned(), "[at the start of an iteration] `self._xyz` is not in pinned memory"
            assert gaussians._scaling.is_pinned(), "[at the start of an iteration] `self._scaling` is not in pinned memory"
            assert gaussians._rotation.is_pinned(), "[at the start of an iteration] `self._rotation` is not in pinned memory"
            assert gaussians._opacity.is_pinned(), "[at the start of an iteration] `self._opacity` is not in pinned memory"
            assert gaussians._features_dc.is_pinned(), "[at the start of an iteration] `self._features_dc` is not in pinned memory"
            assert gaussians._features_rest.is_pinned(), "[at the start of an iteration] `self._features_rest` is not in pinned memory"
        elif args.offload and args.mxw_debug == 'cat':
            assert gaussians._parameters.is_pinned(), "[at the start of an iteration] `self._parameters` is not in pinned memory"
        
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (iteration % args.densification_interval) == 0:
                torch.cuda.memory._record_memory_history()
                log_file.write(
                    "[ITER {}] Tracing cuda memory usage.\n".format(iteration)
                )
        
        # Step Initialization
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)
        num_trained_batches += 1
        # utils.gaussian_report(gaussians)
        # utils.memory_report("at the beginning of an iteration")
        
        # Reset max memory tracking stats
        if args.reset_each_iter:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
        
        timers.clear()
        timers.start("[iteration end2end]")
        if args.nsys_profile:
            assert args.bsz == 1, "nsys profiling only supports batch size 1"
            if iteration == args.nsys_profile_start_iter:
                torch.cuda.cudart().cudaProfilerStart()
            if iteration == args.nsys_profile_end_iter or iteration == opt_args.iterations:
                torch.cuda.cudart().cudaProfilerStop()
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # Prepare data: Pick random Cameras for training
        if args.local_sampling:
            assert (
                args.bsz % utils.WORLD_SIZE == 0
            ), "Batch size should be divisible by the number of GPUs."
            batched_cameras_idx = train_dataset.get_batched_cameras_idx(
                args.bsz // utils.WORLD_SIZE
            )
            batched_all_cameras_idx = torch.zeros(
                (utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int
            )
            batched_cameras_idx = torch.tensor(
                batched_cameras_idx, device="cuda", dtype=int
            )
            torch.distributed.all_gather_into_tensor(
                batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP
            )
            batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
            batched_cameras = train_dataset.get_batched_cameras_from_idx(
                batched_all_cameras_idx
            )
        elif args.torch_dataloader:
            timers.start("dataloader: load the next image from disk and decode")
            try:
                batched_cameras = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batched_cameras = next(dataloader_iter)
            timers.stop("dataloader: load the next image from disk and decode")
            
            #TODO: `camera.uid` should be the id within a batch. Currently we use a postfix as workaround.
            for uid, c in enumerate(batched_cameras):
                c.uid = uid
            
            # Send matrices to gpu
            timers.start("send cam matrices to gpu")
            for camera in batched_cameras:
                camera.world_view_transform = camera.world_view_transform.cuda()
                # camera.projection_matrix = camera.projection_matrix.cuda()
                camera.full_proj_transform = camera.full_proj_transform.cuda()
            timers.stop("send cam matrices to gpu")
            
        else:
            batched_cameras = train_dataset.get_batched_cameras(args.bsz)

        with torch.no_grad():
            # Prepare Workload division strategy
            timers.start("prepare_strategies")
            batched_strategies, gpuid2tasks = start_strategy_final(
                batched_cameras, strategy_history
            )
            timers.stop("prepare_strategies")

            # Load ground-truth images to GPU
            timers.start("load_cameras")
            load_camera_from_cpu_to_all_gpu(
                batched_cameras, batched_strategies, gpuid2tasks
            )
            timers.stop("load_cameras")

        if args.offload and args.bsz > 1:
            assert utils.DEFAULT_GROUP.size() == 1, "Offloading is implemented only for one GPU"
            N = gaussians._xyz.shape[0]
            # Stream the computations and accumulate grads
            for cam_id, (camera, strategy) in enumerate(zip(batched_cameras, batched_strategies)):
                if args.backend == "gsplat":
                    timers.start("preprocess_final")
                    batched_screenspace_pkg = (
                        gsplat_distributed_preprocess3dgs_and_all2all_final(
                            [camera],
                            gaussians,
                            pipe_args,
                            background,
                            batched_strategies=[strategy],
                            mode="train",
                            offload=args.offload,
                            means3D_all=means3D_all,
                            prev_filter=send2gpu_filter,
                            prev_filter_cpu=send2gpu_filter_cpu,
                        )
                    )
                    timers.stop("preprocess_final")
                    
                    if ((iteration + cam_id) % args.log_interval) == 1:
                        num_visible = batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"].sum().item()
                        log_file.write(
                            "<<< # iteration: {}, visible gaussians this iter = {}/{} (%{:.2f}) >>>\n".format((iteration + cam_id), num_visible, gaussians._xyz.shape[0], (100 * num_visible / gaussians._xyz.shape[0]))
                        )
                    
                    # utils.memory_report("after preprocessing")
                    timers.start("render_final")
                    batched_image, batched_compute_locally = gsplat_render_final(
                        batched_screenspace_pkg, [strategy]
                    )
                    timers.stop("render_final")
                    batch_statistic_collector = [
                        cuda_args["stats_collector"]
                        for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                    ]
                    # utils.memory_report("after rendering")
                else:
                    timers.start("preprocess_final")
                    batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                        [camera],
                        gaussians,
                        pipe_args,
                        background,
                        batched_strategies=[strategy],
                        mode="train",
                    )
                    timers.stop("preprocess_final")
                    
                    timers.start("render_final")
                    batched_image, batched_compute_locally = render_final(
                        batched_screenspace_pkg, [strategy]
                    )
                    timers.stop("render_final")
                    batch_statistic_collector = [
                        cuda_args["stats_collector"]
                        for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                    ]
                
                loss_sum, batched_losses = batched_loss_computation(
                    batched_image,
                    [camera],
                    batched_compute_locally,
                    [strategy],
                    batch_statistic_collector,
                )
                
                timers.start("backward")
                loss_sum.backward()
                timers.stop("backward")
                utils.check_initial_gpu_memory_usage("after backward")
                # utils.memory_report("after backward")
                del loss_sum # fix memory leak
                
                with torch.no_grad():
                    if args.offload and args.exact_filter:
                        if args.mxw_debug == 'cat':
                            means3D_all, means3D, opacities, scales, rotations, shs = batched_screenspace_pkg["param_handles"]
                            (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = batched_screenspace_pkg["send2gpu_filter"]

                            # timers.start("zero out grads")
                            # N = gaussians._xyz.shape[0]
                            # parameters_grad_buffer[:N, :].zero_()
                            # timers.stop("zero out grads")

                            timers.start("fused grad transfer")
                            param_dims = torch.tensor([3, 1, 3, 4, 48], device="cuda", dtype=torch.int32)
                            param_dims_presum_rshift = torch.tensor([0, 3, 4, 7, 11], device="cuda", dtype=torch.int32)
                            col2attr = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3] + [4] * 48, device="cuda", dtype=torch.int32)
                            send2cpu_cat_buffer_osr_shs(
                                means3D.grad,
                                opacities.grad,
                                scales.grad,
                                rotations.grad,
                                shs.grad,
                                infrustum_radii_opacities_filter_indices,
                                send2gpu_final_filter_indices,
                                param_dims,
                                param_dims_presum_rshift,
                                col2attr,
                                parameters_grad_buffer[:N, :],
                                accum=True
                            ) # This kernel blocks the cpu.
                            timers.stop("fused grad transfer")

                            # Free grads on gpu
                            means3D.grad = None
                            opacities.grad = None
                            scales.grad = None
                            rotations.grad = None
                            shs.grad = None
                            
                            del means3D, opacities, scales, rotations, shs
                            
                            timers.start("load from buffer")
                            gaussians._parameters.grad = parameters_grad_buffer[:N, :]
                            timers.stop("load from buffer")                
                            pass
                        else:
                            assert False, "Not implemented yet."

                    elif args.offload:
                        timers.start("sync_grad_to_cpu")
                        if args.mxw_debug == 'fused':                    
                            timers.start("get all handles")
                            means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                            send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                            timers.stop("get all handles")
                            
                            # timers.start("zero out grads")
                            N = gaussians._xyz.shape[0]
                            # means3D_grad_buffer[:N, :].zero_()
                            # opacities_grad_buffer[:N, :].zero_()
                            # scales_grad_buffer[:N, :].zero_()
                            # rotations_grad_buffer[:N, :].zero_()
                            # features_dc_grad_buffer[:N, :, :].zero_()
                            # features_rest_grad_buffer[:N, :, :].zero_()
                            # timers.stop("zero out grads")
                            
                            timers.start("fused grad transfer")
                            send2cpu(
                                means3D.grad,
                                opacities.grad,
                                scales.grad,
                                rotations.grad,
                                features_dc.grad,
                                features_rest.grad,
                                send2gpu_filter,
                                means3D_grad_buffer[:N, :],
                                opacities_grad_buffer[:N, :],
                                scales_grad_buffer[:N, :],
                                rotations_grad_buffer[:N, :],
                                features_dc_grad_buffer[:N, :, :],
                                features_rest_grad_buffer[:N, :, :]
                            ) # This kernel blocks the cpu.
                            timers.stop("fused grad transfer")
                            
                            # Free grads on gpu
                            means3D.grad = None
                            opacities.grad = None
                            scales.grad = None
                            rotations.grad = None
                            features_dc.grad = None
                            features_rest.grad = None
                            
                            timers.start("load from buffer")
                            gaussians._xyz.grad = means3D_grad_buffer[:N, :]
                            gaussians._opacity.grad = opacities_grad_buffer[:N, :]
                            gaussians._scaling.grad = scales_grad_buffer[:N, :]
                            gaussians._rotation.grad = rotations_grad_buffer[:N, :]
                            gaussians._features_dc.grad = features_dc_grad_buffer[:N, :, :]
                            gaussians._features_rest.grad = features_rest_grad_buffer[:N, :, :]
                            timers.stop("load from buffer") 
                        elif args.mxw_debug == 'cat':
                            timers.start("get all handles")
                            means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                            send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                            timers.stop("get all handles")
                            
                            timers.start("zero out grads")
                            N = gaussians._xyz.shape[0]
                            parameters_grad_buffer[:N, :].zero_()
                            timers.stop("zero out grads")
                            
                            timers.start("fused grad transfer")
                            send2cpu_cat_buffer(
                                means3D.grad,
                                opacities.grad,
                                scales.grad,
                                rotations.grad,
                                features_dc.grad,
                                features_rest.grad,
                                send2gpu_filter,
                                gaussians.param_dims,
                                gaussians.param_dims_presum_rshift,
                                gaussians.col2attr,
                                parameters_grad_buffer[:N, :],
                            ) # This kernel blocks the cpu.
                            timers.stop("fused grad transfer")
                            
                            # Free grads on gpu
                            means3D.grad = None
                            opacities.grad = None
                            scales.grad = None
                            rotations.grad = None
                            features_dc.grad = None
                            features_rest.grad = None
                            
                            del means3D, opacities, scales, rotations, features_dc, features_rest
                            
                            timers.start("load from buffer")
                            gaussians._parameters.grad = parameters_grad_buffer[:N, :]
                            timers.stop("load from buffer") 
                        else:
                            timers.start("get all handles")
                            means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                            send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                            send2gpu_filter_cpu = batched_screenspace_pkg["send2gpu_filter_cpu"]
                            timers.stop("get all handles")

                            # timers.start("zero init grads")
                            # gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
                            # gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
                            # gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
                            # gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
                            # gaussians._features_dc.grad = torch.zeros_like(gaussians._features_dc)
                            # gaussians._features_rest.grad = torch.zeros_like(gaussians._features_rest)
                            # timers.stop("zero init grads")
                            
                            timers.start("transfer grads to cpu")
                            gaussians._xyz.grad[send2gpu_filter_cpu] = means3D.grad.cpu()
                            gaussians._opacity.grad[send2gpu_filter_cpu] = opacities.grad.cpu()
                            gaussians._scaling.grad[send2gpu_filter_cpu] = scales.grad.cpu()
                            gaussians._rotation.grad[send2gpu_filter_cpu] = rotations.grad.cpu()
                            gaussians._features_dc.grad[send2gpu_filter_cpu] = features_dc.grad.cpu()
                            gaussians._features_rest.grad[send2gpu_filter_cpu] = features_rest.grad.cpu()
                            timers.stop("transfer grads to cpu")

                        timers.stop("sync_grad_to_cpu")
                        # utils.memory_report("after syncing grad to cpu")
                    
                    # Sync losses in the batch
                    timers.start("sync_loss_and_log")
                    batched_losses = torch.tensor(batched_losses, device="cuda")
                    # if utils.DEFAULT_GROUP.size() > 1:
                    #     dist.all_reduce(
                    #         batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                    #     )
                    batched_loss = (1.0 - args.lambda_dssim) * batched_losses[:, 0] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
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
                    log_string = "iteration[{},{})-{}, loss: {} image: {}\n".format(
                        iteration,
                        iteration + args.bsz,
                        iteration + cam_id,
                        batched_loss_cpu,
                        # [viewpoint_cam.image_name for viewpoint_cam in [batched_cameras[0]]],
                        camera.image_name
                    )
                    log_file.write(log_string)
                    timers.stop("sync_loss_and_log")
                    
                    del batched_losses # fix memory leak
                    
                    if args.backend == "gsplat":
                        gsplat_densification(
                            iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload, stat_only=True
                        )
                        if not args.disable_auto_densification and iteration <= args.densify_until_iter and iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                            iteration, args.bsz, args.densification_interval, 0
                        ):
                            means3D_all = None
                            send2gpu_filter = None
                            send2gpu_filter_cpu = None    
                    else:
                        assert False, "Not implemented yet"
                    # utils.memory_report("after densification and before freeing activation states")
                    
                    # Free activation states
                    batched_screenspace_pkg = None
                    # batched_image = None #TODO: find a better way to free the gt
                    batched_compute_locally = None
                                    
        else:
            if args.backend == "gsplat":
                # utils.memory_report("before preprocessing")
                timers.start("preprocess_final")
                batched_screenspace_pkg = (
                    gsplat_distributed_preprocess3dgs_and_all2all_final(
                        batched_cameras,
                        gaussians,
                        pipe_args,
                        background,
                        batched_strategies=batched_strategies,
                        mode="train",
                        offload=args.offload,
                        means3D_all=means3D_all,
                        prev_filter=send2gpu_filter,
                        prev_filter_cpu=send2gpu_filter_cpu,
                    )
                )
                timers.stop("preprocess_final")
                
                if (iteration % args.log_interval) == 1:
                    num_visible = batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"].sum().item()
                    log_file.write(
                        "<<< # iteration: {}, visible gaussians this iter = {}/{} (%{:.2f}) >>>\n".format(iteration, num_visible, gaussians._xyz.shape[0], (100 * num_visible / gaussians._xyz.shape[0]))
                    )
                
                # utils.memory_report("after preprocessing")
                timers.start("render_final")
                batched_image, batched_compute_locally = gsplat_render_final(
                    batched_screenspace_pkg, batched_strategies
                )
                timers.stop("render_final")
                
                batch_statistic_collector = [
                    cuda_args["stats_collector"]
                    for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                ]
                # utils.memory_report("after rendering")
            else:
                timers.start("preprocess_final")
                batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                    batched_cameras,
                    gaussians,
                    pipe_args,
                    background,
                    batched_strategies=batched_strategies,
                    mode="train",
                )
                timers.stop("preprocess_final")
                
                timers.start("render_final")
                batched_image, batched_compute_locally = render_final(
                    batched_screenspace_pkg, batched_strategies
                )
                timers.stop("render_final")
                batch_statistic_collector = [
                    cuda_args["stats_collector"]
                    for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                ]

            loss_sum, batched_losses = batched_loss_computation(
                batched_image,
                batched_cameras,
                batched_compute_locally,
                batched_strategies,
                batch_statistic_collector,
            )

            timers.start("backward")
            loss_sum.backward()
            timers.stop("backward")
            utils.check_initial_gpu_memory_usage("after backward")
            # utils.memory_report("after backward")
            del loss_sum # fix memory leak
            
            with torch.no_grad():
                # Sync grad with cpu.
                if args.offload and args.exact_filter:
                    if args.mxw_debug == 'cat':
                        means3D_all, means3D, opacities, scales, rotations, shs = batched_screenspace_pkg["param_handles"]
                        (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = batched_screenspace_pkg["send2gpu_filter"]

                        timers.start("zero out grads")
                        N = gaussians._xyz.shape[0]
                        parameters_grad_buffer[:N, :].zero_()
                        timers.stop("zero out grads")

                        timers.start("fused grad transfer")
                        param_dims = torch.tensor([3, 1, 3, 4, 48], device="cuda", dtype=torch.int32)
                        param_dims_presum_rshift = torch.tensor([0, 3, 4, 7, 11], device="cuda", dtype=torch.int32)
                        col2attr = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3] + [4] * 48, device="cuda", dtype=torch.int32)
                        send2cpu_cat_buffer_osr_shs(
                            means3D.grad,
                            opacities.grad,
                            scales.grad,
                            rotations.grad,
                            shs.grad,
                            infrustum_radii_opacities_filter_indices,
                            send2gpu_final_filter_indices,
                            param_dims,
                            param_dims_presum_rshift,
                            col2attr,
                            parameters_grad_buffer[:N, :],
                        ) # This kernel blocks the cpu.
                        timers.stop("fused grad transfer")

                        # Free grads on gpu
                        means3D.grad = None
                        opacities.grad = None
                        scales.grad = None
                        rotations.grad = None
                        shs.grad = None
                        
                        del means3D, opacities, scales, rotations, shs
                        
                        timers.start("load from buffer")
                        gaussians._parameters.grad = parameters_grad_buffer[:N, :]
                        timers.stop("load from buffer")                     
                        pass
                    else:
                        assert False, "Not implemented yet."

                elif args.offload:
                    timers.start("sync_grad_to_cpu")
                    if args.mxw_debug == 'fused':                    
                        timers.start("get all handles")
                        means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                        send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                        timers.stop("get all handles")
                        
                        timers.start("zero out grads")
                        N = gaussians._xyz.shape[0]
                        means3D_grad_buffer[:N, :].zero_()
                        opacities_grad_buffer[:N, :].zero_()
                        scales_grad_buffer[:N, :].zero_()
                        rotations_grad_buffer[:N, :].zero_()
                        features_dc_grad_buffer[:N, :, :].zero_()
                        features_rest_grad_buffer[:N, :, :].zero_()
                        timers.stop("zero out grads")
                        
                        timers.start("fused grad transfer")
                        send2cpu(
                            means3D.grad,
                            opacities.grad,
                            scales.grad,
                            rotations.grad,
                            features_dc.grad,
                            features_rest.grad,
                            send2gpu_filter,
                            means3D_grad_buffer[:N, :],
                            opacities_grad_buffer[:N, :],
                            scales_grad_buffer[:N, :],
                            rotations_grad_buffer[:N, :],
                            features_dc_grad_buffer[:N, :, :],
                            features_rest_grad_buffer[:N, :, :]
                        ) # This kernel blocks the cpu.
                        timers.stop("fused grad transfer")
                        
                        # Free grads on gpu
                        means3D.grad = None
                        opacities.grad = None
                        scales.grad = None
                        rotations.grad = None
                        features_dc.grad = None
                        features_rest.grad = None
                        
                        timers.start("load from buffer")
                        gaussians._xyz.grad = means3D_grad_buffer[:N, :]
                        gaussians._opacity.grad = opacities_grad_buffer[:N, :]
                        gaussians._scaling.grad = scales_grad_buffer[:N, :]
                        gaussians._rotation.grad = rotations_grad_buffer[:N, :]
                        gaussians._features_dc.grad = features_dc_grad_buffer[:N, :, :]
                        gaussians._features_rest.grad = features_rest_grad_buffer[:N, :, :]
                        timers.stop("load from buffer") 
                    elif args.mxw_debug == 'cat':
                        timers.start("get all handles")
                        means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                        send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                        timers.stop("get all handles")
                        
                        timers.start("zero out grads")
                        N = gaussians._xyz.shape[0]
                        parameters_grad_buffer[:N, :].zero_()
                        timers.stop("zero out grads")
                        
                        timers.start("fused grad transfer")
                        send2cpu_cat_buffer(
                            means3D.grad,
                            opacities.grad,
                            scales.grad,
                            rotations.grad,
                            features_dc.grad,
                            features_rest.grad,
                            send2gpu_filter,
                            gaussians.param_dims,
                            gaussians.param_dims_presum_rshift,
                            gaussians.col2attr,
                            parameters_grad_buffer[:N, :],
                        ) # This kernel blocks the cpu.
                        timers.stop("fused grad transfer")
                        
                        # Free grads on gpu
                        means3D.grad = None
                        opacities.grad = None
                        scales.grad = None
                        rotations.grad = None
                        features_dc.grad = None
                        features_rest.grad = None
                        
                        del means3D, opacities, scales, rotations, features_dc, features_rest
                        
                        timers.start("load from buffer")
                        gaussians._parameters.grad = parameters_grad_buffer[:N, :]
                        timers.stop("load from buffer") 
                    else:
                        timers.start("get all handles")
                        means3D_all, means3D, opacities, scales, rotations, features_dc, features_rest = batched_screenspace_pkg["param_handles"]
                        send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
                        send2gpu_filter_cpu = batched_screenspace_pkg["send2gpu_filter_cpu"]
                        timers.stop("get all handles")

                        timers.start("zero init grads")
                        gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
                        gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
                        gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
                        gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
                        gaussians._features_dc.grad = torch.zeros_like(gaussians._features_dc)
                        gaussians._features_rest.grad = torch.zeros_like(gaussians._features_rest)
                        timers.stop("zero init grads")
                        
                        timers.start("transfer grads to cpu")
                        gaussians._xyz.grad[send2gpu_filter_cpu] = means3D.grad.cpu()
                        gaussians._opacity.grad[send2gpu_filter_cpu] = opacities.grad.cpu()
                        gaussians._scaling.grad[send2gpu_filter_cpu] = scales.grad.cpu()
                        gaussians._rotation.grad[send2gpu_filter_cpu] = rotations.grad.cpu()
                        gaussians._features_dc.grad[send2gpu_filter_cpu] = features_dc.grad.cpu()
                        gaussians._features_rest.grad[send2gpu_filter_cpu] = features_rest.grad.cpu()
                        timers.stop("transfer grads to cpu")

                    timers.stop("sync_grad_to_cpu")
                    # utils.memory_report("after syncing grad to cpu")
                
                # Adjust workload division strategy.
                globally_sync_for_timer()
                timers.start("finish_strategy_final")
                finish_strategy_final(
                    batched_cameras,
                    strategy_history,
                    batched_strategies,
                    batch_statistic_collector,
                )
                timers.stop("finish_strategy_final")

                # Sync losses in the batch
                timers.start("sync_loss_and_log")
                batched_losses = torch.tensor(batched_losses, device="cuda")
                if utils.DEFAULT_GROUP.size() > 1:
                    dist.all_reduce(
                        batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                    )
                batched_loss = (1.0 - args.lambda_dssim) * batched_losses[
                    :, 0
                ] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
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
                timers.stop("sync_loss_and_log")
                
                del batched_losses # fix memory leak

        with torch.no_grad():
            # Evaluation
            end2end_timers.stop()
            training_report(
                iteration,
                l1_loss,
                args.test_iterations,
                scene,
                pipe_args,
                background,
                args.backend,
                args.offload,
            )
            end2end_timers.start()

            # Densification
            # utils.memory_report("before densification")
            if args.backend == "gsplat":
                if args.offload and args.bsz > 1:
                    gsplat_densification(
                        iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload, densify_only=True
                    )
                else:  
                    gsplat_densification(
                        iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload
                    )
                if not args.disable_auto_densification and iteration <= args.densify_until_iter and iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                    iteration, args.bsz, args.densification_interval, 0
                ):
                    means3D_all = None
                    send2gpu_filter = None
                    send2gpu_filter_cpu = None
                    
            else:
                densification(iteration, scene, gaussians, batched_screenspace_pkg)
            # utils.memory_report("after densification and before freeing activation states")
            
            # Free activation states
            batched_screenspace_pkg = None
            batched_image = None
            batched_compute_locally = None
            # utils.memory_report("after freeing activation states and before saving gaussians")

            # Save Gaussians
            if not args.do_not_save and any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

                if args.save_strategy_history:
                    with open(
                        args.log_folder
                        + "/strategy_history_ws="
                        + str(utils.WORLD_SIZE)
                        + "_rk="
                        + str(utils.GLOBAL_RANK)
                        + ".json",
                        "w",
                    ) as f:
                        json.dump(strategy_history.to_json(), f)
                end2end_timers.start()
                # utils.memory_report("after densification")

            # Save Checkpoints
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
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    if utils.DEFAULT_GROUP.size() > 1:
                        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                elif utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder
                    + "/chkpnt_ws="
                    + str(utils.WORLD_SIZE)
                    + "_rk="
                    + str(utils.GLOBAL_RANK)
                    + ".pth",
                )
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                # utils.memory_report("before optimizer step")
                timers.start("optimizer_step")

                if (
                    args.lr_scale_mode != "accumu"
                ):  # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz

                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timers.stop("optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")
                # utils.memory_report("after optimizer step")
                
                if args.offload and args.exact_filter and args.mxw_debug == 'cat':
                    timers.start("zero out grads")
                    parameters_grad_buffer[:N, :].zero_()
                    timers.stop("zero out grads")
                    

        # Finish a iteration and clean up
        torch.cuda.synchronize()
        for (
            viewpoint_cam
        ) in batched_cameras:  # Release memory of locally rendered original_image
            viewpoint_cam.original_image = None
        if args.nsys_profile:
            if iteration >= args.nsys_profile_start_iter and iteration < args.nsys_profile_end_iter:
                nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.stop("[iteration end2end]")
            timers.printTimers(iteration, mode="sum")
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (iteration % args.densification_interval) == 0:
                dump_name = args.model_path + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            
        utils.memory_report("at the end of the iteration")
        log_file.flush()

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    progress_bar.close()
    scene.clean_up()
    
    if args.nsys_profile:
        torch.cuda.cudart().cudaProfilerStop()


def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend, offload=False
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
                "num_cameras": max(len(scene.getTrainCameras() if scene.getTrainCameras() is not None else scene.getTrainCamerasInfo()) // args.llffhold, args.bsz),
            },
        )

        # init workload division strategy
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                # TODO: if not divisible by world size
                num_cameras = config["num_cameras"] // args.bsz * args.bsz
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    if args.local_sampling:
                        # TODO: if not divisible by world size
                        batched_cameras_idx = eval_dataset.get_batched_cameras_idx(
                            args.bsz // utils.WORLD_SIZE
                        )
                        batched_all_cameras_idx = torch.zeros(
                            (utils.WORLD_SIZE, len(batched_cameras_idx)),
                            device="cuda",
                            dtype=int,
                        )
                        batched_cameras_idx = torch.tensor(
                            batched_cameras_idx, device="cuda", dtype=int
                        )
                        torch.distributed.all_gather_into_tensor(
                            batched_all_cameras_idx,
                            batched_cameras_idx,
                            group=utils.DEFAULT_GROUP,
                        )
                        batched_all_cameras_idx = (
                            batched_all_cameras_idx.cpu().numpy().squeeze()
                        )
                        batched_cameras = eval_dataset.get_batched_cameras_from_idx(
                            batched_all_cameras_idx
                        )
                    else:
                        batched_cameras = eval_dataset.get_batched_cameras(
                            num_camera_to_load
                        )
                    batched_strategies, gpuid2tasks = start_strategy_final(
                        batched_cameras, strategy_history
                    )
                    load_camera_from_cpu_to_all_gpu_for_eval(
                        batched_cameras, batched_strategies, gpuid2tasks
                    )
                    batched_image = []
                    for cam_id, (camera, strategy) in enumerate(zip(batched_cameras, batched_strategies)):
                        #FIXME: quick workaround for verifying the correctness
                        camera.world_view_transform = camera.world_view_transform.cuda()
                        camera.full_proj_transform = camera.full_proj_transform.cuda()
                        
                        if backend == "gsplat":
                            batched_screenspace_pkg = (
                                gsplat_distributed_preprocess3dgs_and_all2all_final(
                                    [camera],
                                    scene.gaussians,
                                    pipe_args,
                                    background,
                                    batched_strategies=[strategy],
                                    mode="test",
                                    offload=offload,
                                )
                            )
                            images, _ = gsplat_render_final(
                                batched_screenspace_pkg, [strategy]
                            )
                            batched_image.append(images[0])
                        else:
                            batched_screenspace_pkg = (
                                distributed_preprocess3dgs_and_all2all_final(
                                    [camera],
                                    scene.gaussians,
                                    pipe_args,
                                    background,
                                    batched_strategies=[strategy],
                                    mode="test",
                                )
                            )
                            images, _ = render_final(
                                batched_screenspace_pkg, [strategy]
                            )
                            batched_image.append(images[0])
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

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(
                            gt_camera.original_image / 255.0, 0.0, 1.0
                        )

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
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
            elif config["cameras_info"] and len(config["cameras_info"]) > 0:
                # Dataset is offloaded to disk
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                # TODO: if not divisible by world size
                num_cameras = config["num_cameras"] // args.bsz * args.bsz
                eval_dataset = TorchSceneDataset(config["cameras"], config["cameras_info"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                # Init dataloader
                if args.num_workers == 0:
                    dataloader = DataLoader(
                        eval_dataset,
                        batch_size=args.bsz,
                        # shuffle=True,
                        drop_last=True,
                        pin_memory=True,
                        collate_fn=custom_collate_fn
                    )
                elif args.num_workers > 0:
                    dataloader = DataLoader(
                        eval_dataset,
                        batch_size=args.bsz,
                        num_workers=args.num_workers,
                        # shuffle=True,
                        drop_last=True,
                        persistent_workers=True,
                        pin_memory=True,
                        collate_fn=custom_collate_fn
                    )
                dataloader_iter = iter(dataloader)
                
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    #FIXME: may have problems when bsz > 1
                    try:
                        batched_cameras = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(dataloader)
                        batched_cameras = next(dataloader_iter)
                    # batched_cameras = eval_dataset.get_batched_cameras(
                    #     num_camera_to_load
                    # )
                    batched_strategies, gpuid2tasks = start_strategy_final(
                        batched_cameras, strategy_history
                    )
                    load_camera_from_cpu_to_all_gpu_for_eval(
                        batched_cameras, batched_strategies, gpuid2tasks
                    )
                    batched_image = []
                    for cam_id, (camera, strategy) in enumerate(zip(batched_cameras, batched_strategies)):
                        #FIXME: quick workaround for verifying the correctness
                        camera.world_view_transform = camera.world_view_transform.cuda()
                        camera.full_proj_transform = camera.full_proj_transform.cuda()
                        
                        if backend == "gsplat":
                            batched_screenspace_pkg = (
                                gsplat_distributed_preprocess3dgs_and_all2all_final(
                                    [camera],
                                    scene.gaussians,
                                    pipe_args,
                                    background,
                                    batched_strategies=[strategy],
                                    mode="test",
                                    offload=offload,
                                )
                            )
                            images, _ = gsplat_render_final(
                                batched_screenspace_pkg, [strategy]
                            )
                            batched_image.append(images[0])
                        else:
                            batched_screenspace_pkg = (
                                distributed_preprocess3dgs_and_all2all_final(
                                    [camera],
                                    scene.gaussians,
                                    pipe_args,
                                    background,
                                    batched_strategies=[strategy],
                                    mode="test",
                                )
                            )
                            images, _ = render_final(
                                batched_screenspace_pkg, [strategy]
                            )
                            batched_image.append(images[0])
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

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(
                            gt_camera.original_image / 255.0, 0.0, 1.0
                        )

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
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
