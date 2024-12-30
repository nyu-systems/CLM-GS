import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
    gsplat_distributed_preprocess3dgs_and_all2all_final,
    gsplat_distributed_preprocess3dgs_and_all2all_offloaded_cacheXYZOSR,
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
    torch_compiled_loss
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import (
    densification, 
    gsplat_densification, 
    update_densification_stats_pipelineoffload_xyzosr,
    update_densification_stats_baseline_accumGrads,
)
from diff_gaussian_rasterization import (
    send2cpu,
    send2cpu_cat_buffer,
    send2cpu_cat_buffer_osr_shs,
    send_shs2cpu_shs_buffer,
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream
)
import torch.multiprocessing
import gc
import math
from gsplat import (
    rasterization,
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)

def calculate_filters(
    batched_cameras,
    xyz_gpu,
    opacity_gpu,
    scaling_gpu,
    rotation_gpu
):
    # calculate filters for all cameras
    filters = []
    with torch.no_grad():
        Ks = []
        viewmats = []
        for i, camera in enumerate(batched_cameras):
            K = camera.create_k_on_gpu()
            viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose # this is originally on gpu
            Ks.append(K)
            viewmats.append(viewmat)
        batched_Ks = torch.stack(Ks)  # (B, 3, 3)
        batched_viewmats = torch.stack(viewmats)  # (B, 4, 4)

        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = (
            fully_fused_projection(
                means=xyz_gpu,
                covars=None,
                quats=rotation_gpu,
                scales=scaling_gpu,
                viewmats=batched_viewmats,
                Ks=batched_Ks,
                width=int(camera.image_width),
                height=int(camera.image_height),
                packed=True,
            )# TODO: this function is too heavy to compute the filters. we can have much cheaper calculation. 
        ) # (B, N), (B, N, 2), (B, N), (B, N, 3), (B, N)

        (
            camera_ids, # (nnz,)
            gaussian_ids, # (nnz,)
            _,
            # radii_packed, # (nnz,)
            _,
            # means2d_packed, # (nnz, 2)
            _,
            # depths_packed, # (nnz,)
            _,
            # conics_packed, # (nnz, 3)
            _,
            # compensations
        ) = proj_results

        output, counts = torch.unique_consecutive(camera_ids, return_counts=True)
        assert torch.all(output == torch.arange(len(batched_cameras)).cuda()), "Here we assume every camera sees at least one gaussian. This error can be caused by the fact that some cameras see no gaussians."
        # TODO: here we assume every camera sees at least one gaussian.
        counts_cpu = counts.cpu().numpy().tolist()
        assert sum(counts_cpu) == gaussian_ids.shape[0], "sum(counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
        gaussian_ids_per_camera = torch.split(gaussian_ids, counts_cpu)

    filters = gaussian_ids_per_camera # on GPU
    return filters, camera_ids, gaussian_ids


def pipeline_forward_one_step(
    filtered_opacity_gpu,
    filtered_scaling_gpu,
    filtered_rotation_gpu,
    filtered_xyz_gpu,
    filtered_shs,
    camera,
    scene,
    gaussians,
    background,
    pipe_args
):
    # print shape of all inputs
    # print("filtered_opacity_gpu shape: ", filtered_opacity_gpu.shape)
    # print("filtered_scaling_gpu shape: ", filtered_scaling_gpu.shape)
    # print("filtered_rotation_gpu shape: ", filtered_rotation_gpu.shape)
    # print("filtered_xyz_gpu shape: ", filtered_xyz_gpu.shape)
    # print("filtered_shs shape: ", filtered_shs.shape)

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    # K = camera.create_k_on_gpu() # create K now, which may invoke cpu-gpu transfer
    K = camera.K
    n_selected = filtered_xyz_gpu.shape[0]
    image_width = int(camera.image_width)
    image_height = int(camera.image_height)
    tile_size = 16
    B = 1 # micro batch size is just 1

    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=filtered_xyz_gpu, # (N, 3)
            covars=None,
            quats=filtered_rotation_gpu,
            scales=filtered_scaling_gpu,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=int(camera.image_width),
            height=int(camera.image_height),
            packed=False,
        )
    ) # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    batched_means2D.retain_grad() # this is only for training. 

    sh_degree = gaussians.active_sh_degree
    camtoworlds = camera.camtoworlds
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    dirs = filtered_xyz_gpu[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs = filtered_shs.reshape(1, n_selected, 16, 3)
    batched_colors = spherical_harmonics(
        degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
    )
    batched_colors = torch.clamp_min(batched_colors + 0.5, 0.0) # (1, N, 3)
    batched_opacities = filtered_opacity_gpu.squeeze(1).unsqueeze(0) # (N, 1) -> (1, N)

    # NOTE: In the above code, we keep the first batch dimension, even if it is always 1. 

    # render
    # Identify intersecting tiles.
    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, B, tile_width, tile_height
    )  # (B, tile_height, tile_width)

    # no need for now
    # global max_num_intersection
    # num_intersection = isect_ids.shape[0]
    # max_num_intersection = max(max_num_intersection, num_intersection)    
    # args = utils.get_args()
    # iteration = utils.get_cur_iter()
    # log_file = utils.get_log_file()
    # if (iteration % args.log_interval) == 1:
    #     log_file.write(
    #         "<<< # iteration: {}, # intersections = {}, max # intersections = {} >>>\n".format(iteration, num_intersection, max_num_intersection)
    #     )

    # TODO: One way to do load balancing: Add two timer operators before and after `rasterize_to_pixels`
    # record_time_start : torch operator(torch.autograd.func)

    # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
    backgrounds = (
        background.repeat(B, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return rendered_image, batched_means2D, batched_radiis

import threading
import queue
import time

def pipeline_offload_impl(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
):
    args = utils.get_args()

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam:
        if order_calculation_version == 0:
            bool_filters = []
            for filter in filters:
                bool_filter = torch.zeros(n_gaussians, dtype=torch.int).cuda()
                bool_filter[filter] = 1
                bool_filters.append(bool_filter)
            overlap_size_matrix = [[0 for _ in range(bsz)] for _ in range(bsz)]
            for i in range(bsz):
                for j in range(i+1, bsz):
                    overlap_size_matrix[i][j] = (bool_filters[i]*bool_filters[j]).sum().item()
                    overlap_size_matrix[j][i] = overlap_size_matrix[i][j]

            ordered_indices = [0]
            default_overlap_size = 0
            our_overlap_size = 0
            for i in range(1, bsz):
                cur_index = ordered_indices[-1]
                next_index = -1
                max_overlap = -1
                for j in range(bsz):
                    if j in ordered_indices:
                        continue
                    overlap = overlap_size_matrix[cur_index][j]
                    if overlap > max_overlap:
                        next_index = j
                        max_overlap = overlap
                ordered_indices.append(next_index)

                default_overlap_size += overlap_size_matrix[i-1][i]
                our_overlap_size += max_overlap

            # Permute cameras and filters to the ordered indices
            batched_cameras = [batched_cameras[i] for i in ordered_indices]
            filters = [filters[i] for i in ordered_indices]
            bool_filters = [bool_filters[i] for i in ordered_indices]

            # Calculate the indices at the beginning of each microbatch. 
            finish_indices_filters = []
            bool_filters_sum = torch.stack(bool_filters, dim=0).sum(dim=0)
            for i in range(bsz+1):
                # get the nonzero indices of bool_filters_sum
                finish_indices_filter = torch.nonzero(bool_filters_sum == 0, as_tuple=False).squeeze(1).to(torch.int32)
                finish_indices_filters.append(finish_indices_filter.cpu()) # int32 numpy array
                bool_filters_sum[finish_indices_filter] = int(1e9) # then it will never be 0. 
                if i < bsz:
                    bool_filters_sum -= bool_filters[i]
                else:
                    assert torch.nonzero(bool_filters_sum == 0, as_tuple=False).shape[0] == 0, "bool_filters_sum should be all nonzero."
            assert sum([x.shape[0] for x in finish_indices_filters]) == n_gaussians, "sum of finish_indices_filters should be equal"
        elif order_calculation_version == 1:
            # Use camera_ids and gaussian_ids to fill in the bool_filters
            bool_filters = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            bool_filters[camera_ids, gaussian_ids] = 1
            # Random sample gaussians_ids
            n_sampled = n_gaussians // (bsz ** 2) // 256 * 256 # The matmul complexity is O(bsz^2 * dim) -> O(n_gaussians)
            sampled_gaussian_ids = torch.randperm(n_gaussians, generator=perm_generator, device="cuda")[:n_sampled]
            # sampled_bool_filters = bool_filters[:, sampled_gaussian_ids]
            sampled_gaussian_ids = sampled_gaussian_ids.to(torch.int64)
            sampled_bool_filters = torch.gather(bool_filters, 1, sampled_gaussian_ids.unsqueeze(0).expand(bsz, -1))
            sampled_bool_filters = sampled_bool_filters.to(torch.float32) / 256  # maybe change to fp16 to save memory and compute
            # Calculate the overlap matrix
            overlap_matrix = torch.matmul(sampled_bool_filters, sampled_bool_filters.T) # bsz x bsz
            assert overlap_matrix.shape == (bsz, bsz), "overlap_matrix should be a square matrix."
            overlap_matrix_cpu = overlap_matrix.cpu().numpy()
            # Calculate the order of cameras
            ordered_indices = [0]
            default_overlap_size = 0
            our_overlap_size = 0
            for i in range(1, bsz):
                cur_index = ordered_indices[-1]
                next_index = -1
                max_overlap = -1
                for j in range(bsz):
                    if j in ordered_indices:
                        continue
                    overlap = overlap_matrix_cpu[cur_index][j]
                    if overlap > max_overlap:
                        next_index = j
                        max_overlap = overlap
                ordered_indices.append(next_index)

                default_overlap_size += overlap_matrix_cpu[i-1][i]
                our_overlap_size += max_overlap # This can be done on gpu with only single warp. 

            # print(f"our overlap size: {our_overlap_size}, default overlap size: {default_overlap_size}") # DEBUG code. to be deleted. 

            # Permute cameras and filters to the ordered indices
            batched_cameras = [batched_cameras[i] for i in ordered_indices]
            filters = [filters[i] for i in ordered_indices]
            ordered_indices_gpu = torch.tensor(ordered_indices, dtype=torch.int64, device="cuda")
            bool_filters = torch.gather(bool_filters, 0, ordered_indices_gpu.unsqueeze(1).expand(-1, n_gaussians))
            bool_filters_prefixsum = torch.cumsum(bool_filters, dim=0)
            # Last calculation position for each gaussian
            last_calculation_position_for_each_gaussian = (bool_filters == 1) & (bool_filters_prefixsum == bool_filters_prefixsum[-1].unsqueeze(0)) # (bsz, n_gaussians)
            last_calc_cameraids, last_calc_gaussianids = torch.nonzero(last_calculation_position_for_each_gaussian,
                                                                       as_tuple=True) # shape: (n_nonzero_positions,) , (n_nonzero_positions,)
            not_touched_gaussian_ids = torch.nonzero(bool_filters_prefixsum[-1] == 0, as_tuple=False).squeeze(1)
            last_calc_nonzero_cameras_ids, last_calc_percamera_counts = torch.unique_consecutive(last_calc_cameraids, return_counts=True)
            last_calc_percamera_counts_tmp = torch.zeros(bsz, dtype=torch.int32, device="cuda")
            last_calc_percamera_counts_tmp[last_calc_nonzero_cameras_ids] = last_calc_percamera_counts.to(torch.int32)
            last_calc_percamera_counts = last_calc_percamera_counts_tmp
            assert last_calc_percamera_counts.shape[0] == bsz, "last_calc_percamera_counts should have bsz elements."
            last_calc_percamera_counts_cpu = last_calc_percamera_counts.cpu().tolist()
            last_calc_percamera_counts_cpu = [not_touched_gaussian_ids.shape[0]] + last_calc_percamera_counts_cpu
            last_calc_gaussianids = torch.cat([not_touched_gaussian_ids, last_calc_gaussianids], dim=0).to(torch.int32)
            last_calc_gaussianids_cpu = last_calc_gaussianids.cpu()
            last_calc_gaussian_ids_per_camera = torch.split(last_calc_gaussianids_cpu, last_calc_percamera_counts_cpu)
            assert sum(last_calc_percamera_counts_cpu) == n_gaussians, "sum(last_calc_percamera_counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
            
            finish_indices_filters = last_calc_gaussian_ids_per_camera
            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
        else:
            raise ValueError("Invalid order calculation version.")

        # Define python thread for computing cpuadam
        def cpuadam_thread(bsz,
                           n_gaussians,
                           microbatch_gradient_send_back_events,
                           thread_sync_signal_events,
                           finish_indices_filters,
                           cpu_adam,
                           parameters,
                           parameters_grad):

            if overlap_cpuadam_version == 0:
                parameters.grad = parameters_grad

                cpu_adam.sparse_adam_inc_step() # this is related to lr. 
                if not args.stop_update_param:
                    cpu_adam.sparse_step(sparse_indices=finish_indices_filters[0], version=2, scale=1.0/bsz)

                for i in range(0, bsz):
                    
                    thread_sync_signal_events[i].wait() # wait for the signal of finishing the i-th microbatch.

                    finish_event = microbatch_gradient_send_back_events[i] # event of finishing i-th micro batch.
                    finish_event.synchronize() # synchronize with the gpu event on computation stream.

                    finish_indices_filter = finish_indices_filters[i+1] # torch int32 array on cpu
                    if not args.stop_update_param and finish_indices_filter.shape[0] > 0: # the finish filter should not be empty
                        cpu_adam.sparse_step(sparse_indices=finish_indices_filter, version=2, scale=1.0/bsz)

                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
            elif overlap_cpuadam_version == 1:
                parameters.grad = parameters_grad / bsz
                cpu_adam.step()
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
            elif overlap_cpuadam_version == 2:
                parameters.grad = parameters_grad / bsz
                cpu_adam.sparse_adam_inc_step()
                for filter in finish_indices_filters:
                    cpu_adam.sparse_step(sparse_indices=filter, version=1, scale=1.0/bsz)
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
            else:
                raise ValueError("Invalid version number for cpuadam_thread.")

        # Create thread for cpuadam
        thread_sync_signal_events = [threading.Event() for _ in range(bsz)]
        main_thread_sync_signal_idx = 0
        microbatch_gradient_send_back_events = [
            torch.cuda.Event() for _ in range(bsz)
        ]
        cpuadam_worker = threading.Thread(target=cpuadam_thread, args=(bsz,
                                                                    n_gaussians,
                                                                    microbatch_gradient_send_back_events,
                                                                    thread_sync_signal_events,
                                                                    finish_indices_filters,
                                                                    gaussians.optimizer.cpu_adam,
                                                                    gaussians._parameters,
                                                                    parameters_grad_buffer[:n_gaussians, :],
                                                                    ))
        if overlap_cpuadam_version == 0:
            cpuadam_worker.start()

    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    opacity_gpu = opacity_gpu_origin.detach().requires_grad_()
    scaling_gpu = scaling_gpu_origin.detach().requires_grad_()
    rotation_gpu = rotation_gpu_origin.detach().requires_grad_()

    # declare streams for computationa and communication
    # comm_stream = torch.cuda.Stream(device=0)
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_grad = None
    grid_size, block_size = args.grid_size, 256
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream):
                # Forward pass
                shs = torch.empty(filters[micro_idx].shape[0], 48, device="cuda", requires_grad=True)
                # print("shs shape: ", shs.shape)
                # print("_parameters shape: ", gaussians._parameters.shape)
                # print("filters shape: ", filters[micro_idx].shape)
                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            cpu2gpu_event = next_cpu2gpu_event

        with torch.cuda.stream(comm_stream):
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda", requires_grad=True)
                # print("shs shape: ", shs_next.shape)
                # print("_parameters shape: ", gaussians._parameters.shape)
                # print("filters shape: ", filters[micro_idx+1].shape)
                send_shs2gpu_stream(
                    shs_next,
                    gaussians._parameters,
                    filters[micro_idx + 1],
                    grid_size, block_size
                )
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        with torch.cuda.stream(comm_stream):
            last_microbatch_shs_grad = shs_grad
            shs_grad = torch.empty_like(shs)

        if args.offload_shs_grad_before_every_microbatch and micro_idx > 0:
            with torch.cuda.stream(comm_stream):
                gpu2cpu_event.wait(comm_stream)
                # sync event of default_stream with comm_stream
                # timers.start("fused grad transfer") # rewrite the timer function.
                send_shs2cpu_grad_buffer_stream(
                    last_microbatch_shs_grad,
                    parameters_grad_buffer[:N, :],
                    filters[micro_idx-1],
                    True,
                    grid_size, block_size
                )
                if args.overlap_cpuadam:
                    event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                    event.record()
                    thread_sync_signal_events[main_thread_sync_signal_idx].set()
                    main_thread_sync_signal_idx += 1

        torch.cuda.nvtx.range_push("forward_pass")
        this_filter = filters[micro_idx]
        filtered_xyz_gpu = torch.gather(xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3))
        filtered_opacity_gpu = torch.gather(opacity_gpu, 0, this_filter.reshape(-1, 1))
        filtered_scaling_gpu = torch.gather(scaling_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3))
        filtered_rotation_gpu = torch.gather(rotation_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 4))
        # sync event of comm_stream with default_stream to make sure shs has been loaded to gpu
        cpu2gpu_event.wait(default_stream)
        filtered_shs = shs.requires_grad_() # this is a view of the original shs.

        # preprocess
        rendered_image, batched_means2D, batched_radiis = pipeline_forward_one_step(filtered_opacity_gpu,
                                                filtered_scaling_gpu,
                                                filtered_rotation_gpu,
                                                filtered_xyz_gpu,
                                                filtered_shs,
                                                batched_cameras[micro_idx],
                                                scene,
                                                gaussians,
                                                background,
                                                pipe_args)

        loss = torch_compiled_loss(rendered_image, batched_cameras[micro_idx].original_image)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward_pass")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        # move shs.grad into shs_grad
        shs_grad.copy_(shs.grad)

        losses.append(loss.detach())

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        if not args.offload_shs_grad_before_every_microbatch:
            with torch.cuda.stream(comm_stream):
                gpu2cpu_event.wait(comm_stream)
                # sync event of default_stream with comm_stream
                # timers.start("fused grad transfer") # rewrite the timer function.
                send_shs2cpu_grad_buffer_stream(
                    shs_grad,
                    parameters_grad_buffer[:N, :],
                    filters[micro_idx],
                    True,
                    grid_size, block_size,
                )
                if args.overlap_cpuadam:
                    event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                    event.record()
                    thread_sync_signal_events[main_thread_sync_signal_idx].set()
                    main_thread_sync_signal_idx += 1

                # timers.stop("fused grad transfer")

                # TODO: Free grads on gpu, I am not sure whether this is safe or not. 
                # We need to double check this. 
                # shs.grad = None

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(batched_cameras[micro_idx].image_height),
            int(batched_cameras[micro_idx].image_width),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        del batched_means2D, batched_radiis

    if args.offload_shs_grad_before_every_microbatch:
        with torch.cuda.stream(comm_stream):
            gpu2cpu_event.wait(comm_stream)
            # sync event of default_stream with comm_stream
            # timers.start("fused grad transfer") # rewrite the timer function.
            send_shs2cpu_grad_buffer_stream(
                shs_grad,
                parameters_grad_buffer[:N, :],
                filters[-1],
                True,
                16, 256,
            )
            if args.overlap_cpuadam:
                event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                event.record()
                thread_sync_signal_events[main_thread_sync_signal_idx].set()
                main_thread_sync_signal_idx += 1

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        assert main_thread_sync_signal_idx == bsz, "main_thread_sync_signal_idx should be equal to bsz."
        if overlap_cpuadam_version != 0:
            cpuadam_worker.start()
        assert args.lr_scale_mode == "sqrt", "Overlap CPUAdam only supports sqrt lr scaling"
        assert args.gpu_cache == "xyzosr", "Overlap CPUAdam only supports xyzosr cache"
        assert not args.stop_update_param, "Overlap CPUAdam does not support stop_update_param"
        # only perform gpu adam
        for param in gaussians.all_parameters()[:4]: # the first 4 parameters are on gpu
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.gpu_adam.step()
        gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)
        cpuadam_worker.join()
    else:
        torch.cuda.synchronize() # we need to make sure gradients have all been sent back to cpu. 
        gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()

    torch.cuda.synchronize()
    return losses

def baseline_accumGrads_micro_step(
    means3D,
    opacities,
    scales,
    rotations,
    shs,
    sh_degree,
    camera,
    background,
    mode="train",
    tile_size=16,
):
    # Prepare camera param.
    image_width = int(camera.image_width)
    image_height = int(camera.image_height)
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    focal_length_x = image_width / (2 * tanfovx)
    focal_length_y = image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, image_width / 2.0],
            [0, focal_length_y, image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    viewmat = camera.world_view_transform.transpose(0, 1)

    assert K.shape == (3, 3)

    radiis, means2D, depths, conics, _ = fully_fused_projection(
        means=means3D,  # (N, 3)
        covars=None,
        quats=rotations,
        scales=scales,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=image_width,
        height=image_height,
        packed=False,
    )

    if mode == "train":
        means2D.retain_grad()
    
    camtoworld = torch.inverse(viewmat.unsqueeze(0))
    dirs = means3D[None, :, :] - camtoworld[:, None, :3, 3]

    colors = spherical_harmonics(
        degrees_to_use=sh_degree,
        dirs=dirs,
        coeffs=shs.unsqueeze(0),
        masks=(radiis > 0)
    )
    colors = torch.clamp_min(colors + 0.5, 0.0)

    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))

    _, isect_ids, flatten_ids = isect_tiles(
        means2d=means2D,
        radii=radiis,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    rendered_image, _ = rasterize_to_pixels(
        means2d=means2D,
        conics=conics,
        colors=colors,
        opacities=opacities.squeeze(1).unsqueeze(0),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=background,
    )
    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return rendered_image, means2D, radiis

def baseline_accumGrads_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    scaling_modifier=1.0
):
    losses = []

    means3D = gaussians.get_xyz
    opacities_origin = gaussians.get_opacity
    scales_origin = gaussians.get_scaling * scaling_modifier
    rotations_origin = gaussians.get_rotation
    shs_origin = gaussians.get_features  # (N, K, 3)
    sh_degree = gaussians.active_sh_degree

    opacities = opacities_origin.detach().requires_grad_(True)
    scales = scales_origin.detach().requires_grad_(True)
    rotations = rotations_origin.detach().requires_grad_(True)
    shs = shs_origin.detach().requires_grad_(True)

    for micro_idx, camera in enumerate(batched_cameras):

        rendered_image, means2D, radiis = baseline_accumGrads_micro_step(
            means3D,
            opacities,
            scales,
            rotations,
            shs,
            sh_degree,
            camera,
            background,
        )
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        loss.backward()
        losses.append(loss.detach())

        with torch.no_grad():
            # Update densification state.
            update_densification_stats_baseline_accumGrads(
                scene,
                gaussians,
                int(camera.image_height),
                int(camera.image_width),
                means2D.grad,
                radiis,
            )
        
        del loss, rendered_image, means2D, radiis
    
    opacities_origin.backward(opacities.grad)
    scales_origin.backward(scales.grad)
    rotations_origin.backward(rotations.grad)
    shs_origin.backward(shs.grad)

    return losses

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1
    if args.sharing_strategy != "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    
    # gc.set_debug(gc.DEBUG_LEAK)

    # Init parameterized scene
    gaussians = GaussianModel(sh_degree=dataset_args.sh_degree)

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
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
        elif args.num_workers > 0:
            dataloader = DataLoader(
                train_dataset,
                batch_size=args.bsz,
                num_workers=args.num_workers,
                shuffle=True,
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
    # if args.offload:
    #     if args.gpu_cache == "xyzosr":
    #         parameters_buffer = torch.empty((args.prealloc_capacity, 48), dtype=torch.float32, pin_memory=True)
    #         parameters_grad_buffer = torch.zeros((args.prealloc_capacity, 48), dtype=torch.float32, pin_memory=True)
    #     elif args.gpu_cache == "no_cache":
    #         parameters_buffer = torch.empty((args.prealloc_capacity, 59), dtype=torch.float32, pin_memory=True)
    #         parameters_grad_buffer = torch.zeros((args.prealloc_capacity, 59), dtype=torch.float32, pin_memory=True)
    #     else:
    #         raise ValueError("Invalid gpu cache strategy.")

    # declare stream for communication
    comm_stream = torch.cuda.Stream(device=0, priority=args.comm_stream_priority)

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

    if args.manual_gc:
        gc.disable()
        gc.collect()


    perm_generator = torch.Generator(device="cuda")
    perm_generator.manual_seed(1)

    ema_loss_for_log = 0
    means3D_all = None # A handle to means3D_all on gpu
    send2gpu_filter = None # A handle to send2gpu_filter on gpu
    send2gpu_filter_cpu = None # A handle to send2gpu_filter on cpu
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):    
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
            # assert args.bsz == 1, "nsys profiling only supports batch size 1"
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
        if args.torch_dataloader:
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

            if args.pipelined_offload:
                batched_world_view_transform = []
                for camera in batched_cameras:
                    camera.K = camera.create_k_on_gpu()
                    batched_world_view_transform.append(camera.world_view_transform.transpose(0, 1))
                batched_world_view_transform = torch.stack(batched_world_view_transform)
                batched_world_view_transform_inverse = torch.inverse(batched_world_view_transform)
                batched_world_view_transform_inverse = torch.unbind(batched_world_view_transform_inverse, dim=0)
                for camera, wvt in zip(batched_cameras, batched_world_view_transform_inverse):
                    camera.camtoworlds = wvt.unsqueeze(0)
                # TODO: maybe we can save them on gpu during initialization. After all, they do not take up lots of memory.
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

        if args.pipelined_offload:
            assert args.offload, "Pipelined offload requires offloading"
            assert args.bsz > 1, "Pipelined offload requires batch size > 1"
            assert args.gpu_cache == "xyzosr", "Pipelined offload requires xyzosr cache"

            N = gaussians._xyz.shape[0]
            losses = pipeline_offload_impl(
                gaussians,
                scene,
                batched_cameras,
                gaussians.parameters_grad_buffer,
                background,
                pipe_args,
                comm_stream,
                perm_generator
            )
            batched_screenspace_pkg = {}

            
            # Sync losses in the batch
            timers.start("sync_loss_and_log")
            batched_losses = torch.stack(losses)
            batched_loss_cpu = batched_losses.cpu().numpy()
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            # Update Epoch Statistics
            train_dataset.update_losses(batched_loss_cpu)
            # Logging
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}), loss: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)

        elif args.offload:
            assert utils.DEFAULT_GROUP.size() == 1, "Offloading is implemented only for one GPU"
            N = gaussians._xyz.shape[0]
            
            # Stream the computations and accumulate grads
            for cam_id, (camera, strategy) in enumerate(zip(batched_cameras, batched_strategies)):
                if args.gpu_cache == "xyzosr":
                    timers.start("preprocess_final")
                    batched_screenspace_pkg = (
                        gsplat_distributed_preprocess3dgs_and_all2all_offloaded_cacheXYZOSR(
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
                    
                    timers.start("render_final")
                    batched_image, batched_compute_locally = gsplat_render_final(
                        batched_screenspace_pkg, [strategy]
                    )
                    timers.stop("render_final")
                    batch_statistic_collector = [
                        cuda_args["stats_collector"]
                        for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                    ]                    
                
                elif args.gpu_cache == "no_cache":
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
                    
                    timers.start("render_final")
                    batched_image, batched_compute_locally = gsplat_render_final(
                        batched_screenspace_pkg, [strategy]
                    )
                    timers.stop("render_final")
                    batch_statistic_collector = [
                        cuda_args["stats_collector"]
                        for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                    ]
                
                else:
                    raise Exception("Invalid gpu cache strategy.")
                
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
                    timers.start("sync_grad_to_cpu")
                    if args.gpu_cache == "xyzosr":
                        (shs, ) = batched_screenspace_pkg["param_handles"]
                        (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = batched_screenspace_pkg["send2gpu_filter"]
                        
                        timers.start("fused grad transfer")
                        send_shs2cpu_shs_buffer(
                            shs.grad,
                            send2gpu_final_filter_indices,
                            gaussians.parameters_grad_buffer[:N, :],
                            accum=True
                        ) # This kernel blocks the cpu.
                        timers.stop("fused grad transfer")

                        # Free grads on gpu
                        shs.grad = None
                        del shs
                        
                        timers.start("load from buffer")
                        gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]
                        timers.stop("load from buffer")
                    
                    elif args.gpu_cache == "no_cache":
                        means3D_all, means3D, opacities, scales, rotations, shs = batched_screenspace_pkg["param_handles"]
                        (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = batched_screenspace_pkg["send2gpu_filter"]

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
                            gaussians.parameters_grad_buffer[:N, :],
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
                        gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]
                        timers.stop("load from buffer")
                    
                    else:
                        raise Exception("Invalid gpu cache strategy.")
                    timers.stop("sync_grad_to_cpu")
                    
                    
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
                    
                    # Densification: update densification stats
                    gsplat_densification(
                        iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload, stat_only=True
                    )
                    if not args.disable_auto_densification and iteration <= args.densify_until_iter and iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                        iteration, args.bsz, args.densification_interval, 0
                    ):
                        means3D_all = None
                        send2gpu_filter = None
                        send2gpu_filter_cpu = None 
                    
                    # Free activation states
                    batched_screenspace_pkg = None
                    # batched_image = None #TODO: find a better way to free the gt
                    batched_compute_locally = None
        elif args.accumulate_grads:
            assert args.backend == "gsplat"
                
            losses = baseline_accumGrads_impl(
                gaussians,
                scene,
                batched_cameras,
                background,
            )
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
            batched_screenspace_pkg = {}
                    
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
                        offload=False,
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
                    batched_screenspace_pkg, batched_strategies, accumulate_grads=args.accumulate_grads
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
                if args.offload:
                    gsplat_densification(
                        iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload, densify_only=True
                    )
                elif args.accumulate_grads:
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
            if iteration < opt_args.iterations and not args.pipelined_offload:
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
                
                if args.offload:
                    timers.start("zero out grads")
                    gaussians.parameters_grad_buffer[:N, :].zero_()
                    timers.stop("zero out grads")
                    

        # Finish a batch and clean up
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
            if args.manual_gc:
                gc.collect()
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (iteration % args.densification_interval) == 0:
                dump_name = args.model_path + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            
        utils.memory_report("at the end of the iteration")
        log_file.flush()
        
        # gc.collect()
        # Dump the garbage collection statistics to a JSON file
        # gc_stats = gc.get_stats()
        # with open('gc_stats.json', 'a') as file:
        #     json.dump(gc_stats, file, indent=4)

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
    
    # HACK
    parameters = gaussians._parameters[::5000].cpu().detach().numpy().tolist()
    json.dump(parameters, open(os.path.join(args.model_path, "shs_parameters.json"), "w"))



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
                "num_cameras": max(len(scene.getTrainCameras() if scene.getTrainCameras() is not None else scene.getTrainCamerasInfo()) // args.llffhold, 1),
            },
        )

        # init workload division strategy
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                # TODO: if not divisible by world size
                num_cameras = config["num_cameras"]
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                for idx in range(1, num_cameras + 1, 1):
                    num_camera_to_load = min(1, num_cameras - idx + 1)
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
                        
                        if args.offload:
                            if args.gpu_cache == "xyzosr":
                                batched_screenspace_pkg = (
                                    gsplat_distributed_preprocess3dgs_and_all2all_offloaded_cacheXYZOSR(
                                        [camera],
                                        scene.gaussians,
                                        pipe_args,
                                        background,
                                        batched_strategies=[strategy],
                                        mode="test",
                                        offload=args.offload,
                                    )
                                )
                                images, _ = gsplat_render_final(
                                    batched_screenspace_pkg, [strategy]
                                )
                                
                                batched_image.append(images[0])
                                del batched_screenspace_pkg         
                            
                            elif args.gpu_cache == "no_cache":
                                batched_screenspace_pkg = (
                                    gsplat_distributed_preprocess3dgs_and_all2all_final(
                                        [camera],
                                        scene.gaussians,
                                        pipe_args,
                                        background,
                                        batched_strategies=[strategy],
                                        mode="test",
                                        offload=args.offload,
                                    )
                                )
                                images, _ = gsplat_render_final(
                                    batched_screenspace_pkg, [strategy]
                                )
                                
                                batched_image.append(images[0])
                                del batched_screenspace_pkg
                            
                            else:
                                raise Exception("Invalid gpu cache strategy.")
                            
                        else:
                            if args.backend == "gsplat":
                                batched_screenspace_pkg = (
                                    gsplat_distributed_preprocess3dgs_and_all2all_final(
                                        [camera],
                                        scene.gaussians,
                                        pipe_args,
                                        background,
                                        batched_strategies=[strategy],
                                        mode="test",
                                        offload=False,
                                    )
                                )
                                images, _ = gsplat_render_final(
                                    batched_screenspace_pkg, [strategy]
                                )
                                
                                batched_image.append(images[0])
                                del batched_screenspace_pkg
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
                                del batched_screenspace_pkg
                        
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
                num_cameras = config["num_cameras"]
                eval_dataset = TorchSceneDataset(config["cameras"], config["cameras_info"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                # Init dataloader: num_workers = 0
                dataloader = DataLoader(
                    eval_dataset,
                    batch_size=1,
                    # shuffle=True,
                    pin_memory=True,
                    collate_fn=custom_collate_fn
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

                        if args.offload:
                            if args.gpu_cache == "xyzosr":
                                batched_screenspace_pkg = (
                                    gsplat_distributed_preprocess3dgs_and_all2all_offloaded_cacheXYZOSR(
                                        [camera],
                                        scene.gaussians,
                                        pipe_args,
                                        background,
                                        batched_strategies=[strategy],
                                        mode="test",
                                        offload=args.offload
                                    )
                                )
                                images, _ = gsplat_render_final(
                                    batched_screenspace_pkg, [strategy]
                                )
                                batched_image.append(images[0])
                                del batched_screenspace_pkg
                            elif args.gpu_cache == "no_cache":
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
                                del batched_screenspace_pkg
                            else:
                                raise ValueError("Invalid gpu cache strategy.")  
                        else:
                            if args.backend == "gsplat":
                                batched_screenspace_pkg = (
                                    gsplat_distributed_preprocess3dgs_and_all2all_final(
                                        [camera],
                                        scene.gaussians,
                                        pipe_args,
                                        background,
                                        batched_strategies=[strategy],
                                        mode="test",
                                        offload=False,
                                    )
                                )
                                images, _ = gsplat_render_final(
                                    batched_screenspace_pkg, [strategy]
                                )
                                
                                batched_image.append(images[0])
                                del batched_screenspace_pkg
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
                                del batched_screenspace_pkg

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
