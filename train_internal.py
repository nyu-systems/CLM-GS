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
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    send_shs2gpu_stream_retention2_64,
    send_shs2cpu_grad_buffer_stream_retention2_64,
)
import diff_gaussian_rasterization
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
    spherical_harmonics_bwd_inplace,
)
from functools import reduce
import fast_tsp

def calculate_filters(
    batched_cameras,
    xyz_gpu,
    opacity_gpu,
    scaling_gpu,
    rotation_gpu
):
    args = utils.get_args()
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
                radius_clip=args.radius_clip,
                width=int(utils.get_img_width()),
                height=int(utils.get_img_height()),
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
    pipe_args,
    eval=False,
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
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
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
            width=image_width,
            height=image_height,
            packed=False,
        )
    ) # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    if not eval:
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

def pipeline_forward_one_step_shs_inplace(
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

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    # K = camera.create_k_on_gpu() # create K now, which may invoke cpu-gpu transfer
    K = camera.K
    n_selected = filtered_xyz_gpu.shape[0]
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
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
            width=image_width,
            height=image_height,
            packed=False,
        )
    ) # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    batched_means2D.retain_grad() # this is only for training. 

    sh_degree = gaussians.active_sh_degree
    camtoworlds = camera.camtoworlds
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    dirs = filtered_xyz_gpu[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs = filtered_shs.reshape(1, n_selected, 16, 3)
    
    with torch.no_grad():
        batched_colors_origin = spherical_harmonics(
            degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
        )
    batched_colors_detached = batched_colors_origin.detach().requires_grad_()
    batched_colors = torch.clamp_min(batched_colors_detached + 0.5, 0.0) # (1, N, 3)
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

    return rendered_image, batched_means2D, batched_radiis, batched_colors_detached, dirs

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
    iteration = utils.get_cur_iter()

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    torch.cuda.nvtx.range_push("calculate_filters")
    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)
    torch.cuda.nvtx.range_pop()

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam:
        torch.cuda.nvtx.range_push("sort cameras")
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
            last_calc_gaussianids_cpu = last_calc_gaussianids.cpu() # TODO: this is slow
            last_calc_gaussian_ids_per_camera = torch.split(last_calc_gaussianids_cpu, last_calc_percamera_counts_cpu)
            assert sum(last_calc_percamera_counts_cpu) == n_gaussians, "sum(last_calc_percamera_counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
            
            finish_indices_filters = last_calc_gaussian_ids_per_camera
            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
        elif order_calculation_version == 2:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            
            not_touched_ids = torch.nonzero(reduce(torch.bitwise_or, torch.unbind(gs_bitmap)) == 0).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = [cur_cam]
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            for i in range(1, bsz):
                cur_cam = ordered_cams[-1]
                gs_bitmap[:, filters[cur_cam]] = 0
                s_vec = torch.sum(gs_bitmap, dim=1, dtype=torch.int32)
                s_vec[ordered_cams] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(s_vec).item()
                ordered_cams.append(next_cam)
                next_update = torch.nonzero(gs_bitmap[next_cam, :]).flatten()
                update_ls.append(next_update)
            update_ls.append(not_touched_ids)
            update_ls.reverse()
            ordered_cams.reverse()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls = cat_update_ls.cpu()
            update_ls_cpu = torch.split(cat_update_ls, update_ls_dim, dim=0)

            finish_indices_filters = update_ls_cpu
            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians

        else:
            raise ValueError("Invalid order calculation version.")

        # Define python thread for computing cpuadam
        def cpuadam_thread_v0(bsz,
                              n_gaussians,
                              microbatch_gradient_send_back_events,
                              thread_sync_signal_events,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")
            parameters.grad = parameters_grad

            cpu_adam.sparse_adam_inc_step() # this is related to lr. 
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.sparse_step(sparse_indices=finish_indices_filters[0], version=2, scale=1.0/bsz)
                torch.cuda.nvtx.range_pop()

            for i in range(0, bsz):
                torch.cuda.nvtx.range_push(f"cpuadam microbatch: {i} - [{iteration},{iteration+bsz})")
                
                thread_sync_signal_events[i].wait() # wait for the signal of finishing the i-th microbatch.

                finish_event = microbatch_gradient_send_back_events[i] # event of finishing i-th micro batch.
                finish_event.synchronize() # synchronize with the gpu event on computation stream.

                finish_indices_filter = finish_indices_filters[i+1] # torch int32 array on cpu
                if not args.stop_update_param and finish_indices_filter.shape[0] > 0: # the finish filter should not be empty
                    cpu_adam.sparse_step(sparse_indices=finish_indices_filter, version=2, scale=1.0/bsz)
                
                torch.cuda.nvtx.range_pop()

            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
            torch.cuda.nvtx.range_pop()

        def cpuadam_thread_v1(bsz,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            parameters.grad = parameters_grad / bsz
            cpu_adam.step()
            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 

        def cpuadam_thread_v2(bsz,
                           finish_indices_filters,
                           cpu_adam,
                           parameters,
                           parameters_grad):
            # parameters.grad = parameters_grad / bsz
            parameters.grad = parameters_grad
            cpu_adam.sparse_adam_inc_step()
            for filter in finish_indices_filters:
                cpu_adam.sparse_step(sparse_indices=filter, version=1, scale=1.0/bsz)
            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 

        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        def cpuadam_thread_v4(bsz,
                              microbatch_gradient_send_back_events,
                              thread_sync_signal_events,
                              signal_tensor_pinned):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")
            for i in range(0, bsz):
                torch.cuda.nvtx.range_push(f"cpuadam microbatch: {i} - [{iteration},{iteration+bsz})")
                
                thread_sync_signal_events[i].wait() # wait for the signal of finishing the i-th microbatch.

                finish_event = microbatch_gradient_send_back_events[i] # event of finishing i-th micro batch.
                finish_event.synchronize() # synchronize with the gpu event on computation stream.

                signal_tensor_pinned[i] = 1 # set signal to 1 to notify gradients have been sent back to cpu

                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

        # Create thread for cpuadam
        if overlap_cpuadam_version == 0:
            thread_sync_signal_events = [threading.Event() for _ in range(bsz)]
            main_thread_sync_signal_idx = 0
            microbatch_gradient_send_back_events = [
                torch.cuda.Event() for _ in range(bsz)
            ]
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v0, args=(bsz,
                                                                        n_gaussians,
                                                                        microbatch_gradient_send_back_events,
                                                                        thread_sync_signal_events,
                                                                        finish_indices_filters,
                                                                        gaussians.optimizer.cpu_adam,
                                                                        gaussians._parameters,
                                                                        parameters_grad_buffer[:n_gaussians, :],
                                                                        ))
            cpuadam_worker.start()
        elif overlap_cpuadam_version == 1:
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v1, args=(bsz,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
        elif overlap_cpuadam_version == 2:
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v2, args=(bsz,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
        elif overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        elif overlap_cpuadam_version == 4:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
            
            thread_sync_signal_events = [threading.Event() for _ in range(bsz)]
            main_thread_sync_signal_idx = 0
            microbatch_gradient_send_back_events = [
                torch.cuda.Event() for _ in range(bsz)
            ]
            cpuadam_synchronization_worer = threading.Thread(target=cpuadam_thread_v4, args=(bsz,
                                                                                             microbatch_gradient_send_back_events,
                                                                                             thread_sync_signal_events,
                                                                                             signal_tensor_pinned))
            cpuadam_synchronization_worer.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
        
        torch.cuda.nvtx.range_pop()

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
    grid_size, block_size = args.grid_size_H, 256
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
                    if args.overlap_cpuadam_version in [0, 4]:
                        event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                        event.record()
                        thread_sync_signal_events[main_thread_sync_signal_idx].set()
                        main_thread_sync_signal_idx += 1
                    elif args.overlap_cpuadam_version == 3:
                        # set signal to pinned memory to notify gradients have been sent back to cpu
                        diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                        microbatch_idx += 1

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

        # # free shs
        # shs.grad = None
        # del shs

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
                    if args.overlap_cpuadam_version in [0, 4]:
                        event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                        event.record()
                        thread_sync_signal_events[main_thread_sync_signal_idx].set()
                        main_thread_sync_signal_idx += 1
                    elif args.overlap_cpuadam_version == 3:
                        # set signal to pinned memory to notify gradients have been sent back to cpu
                        diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                        microbatch_idx += 1


                # timers.stop("fused grad transfer")

                # TODO: Free grads on gpu, I am not sure whether this is safe or not. 
                # We need to double check this. 
                # shs.grad = None

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
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
                if args.overlap_cpuadam_version in [0, 4]:
                    event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                    event.record()
                    thread_sync_signal_events[main_thread_sync_signal_idx].set()
                    main_thread_sync_signal_idx += 1
                elif args.overlap_cpuadam_version == 3:
                    # set signal to pinned memory to notify gradients have been sent back to cpu
                    diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        if args.overlap_cpuadam_version in [0, 4]:
            assert main_thread_sync_signal_idx == bsz, "main_thread_sync_signal_idx should be equal to bsz."
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
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

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses

def pipeline_offload_retention_impl(
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
    iteration = utils.get_cur_iter()
    assert args.offload_shs_grad_before_every_microbatch, "retention currently requires offload_shs_grad_before_every_microbatc"

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    torch.cuda.nvtx.range_push("calculate_filters")
    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)
    torch.cuda.nvtx.range_pop()

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        if order_calculation_version == 2:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            retent_bitmap = gs_bitmap.clone()
            # the following cumsum is slow (14 ms on 10m rubble)
            retent_index = torch.cumsum(retent_bitmap, dim=1, dtype=torch.int64) - 1
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = [cur_cam]
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-1]
                gs_bitmap[:, filters[cur_cam]] = 0
                s_vec = torch.sum(gs_bitmap, dim=1, dtype=torch.int32)
                s_vec[ordered_cams] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(s_vec).item()
                ordered_cams.append(next_cam)
                next_update = torch.nonzero(gs_bitmap[next_cam, :]).flatten()
                update_ls.append(next_update)
            update_ls.append(not_touched_ids)
            update_ls.reverse()
            ordered_cams.reverse()
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            # calculate filters to index retent shs on gpu and shs to load from cpu
            host_indices_to_param = [None]
            param_indices_from_host = [None]
            rtnt_indices_to_param = [None]
            param_indices_from_rtnt = [None]

            # for bwd, since we already have a mapping between device param and device retent, 
            # we just need to add a mapping from device grad to host grad
            host_indices_from_grad = [None]
            grad_indices_to_host = [None]

            torch.cuda.nvtx.range_push("mask calculation loop")
            for i in range(1, bsz):
                curr_cam = ordered_cams[i]
                prev_cam = ordered_cams[i-1]
                curr_idx = retent_index[curr_cam]
                prev_idx = retent_index[prev_cam]

                idx_h = torch.nonzero(retent_bitmap[curr_cam] & ~retent_bitmap[prev_cam]).flatten()
                host_indices_to_param.append(idx_h)
                param_indices_from_host.append(torch.gather(curr_idx, dim=0, index=idx_h))

                idx_d = torch.nonzero(retent_bitmap[curr_cam] & retent_bitmap[prev_cam]).flatten() # overlap
                rtnt_indices_to_param.append(torch.gather(prev_idx, dim=0, index=idx_d))
                param_indices_from_rtnt.append(torch.gather(curr_idx, dim=0, index=idx_d))

                idx_g = torch.nonzero(retent_bitmap[prev_cam] & ~retent_bitmap[curr_cam]).flatten()
                host_indices_from_grad.append(idx_g)
                grad_indices_to_host.append(torch.gather(prev_idx, dim=0, index=idx_g))
            torch.cuda.nvtx.range_pop()

            rtnt_indices_from_grad = param_indices_from_rtnt
            grad_indices_to_rtnt = rtnt_indices_to_param

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls = cat_update_ls.cpu()
            update_ls_cpu = torch.split(cat_update_ls, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians

        elif order_calculation_version == 3:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("cumsum - 1")
            retent_index = torch.empty_like(gs_bitmap, dtype=torch.int64)
            for i in range(bsz):
                retent_index[i] = torch.cumsum(gs_bitmap[i], dim=0, dtype=torch.int64) - 1
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[0] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[i - 1]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[:i].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[i] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.flip(ordered_cams, dims=[0])
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            # calculate filters to index retent shs on gpu and shs to load from cpu
            host_indices_to_param = [None]
            param_indices_from_host = [None]
            rtnt_indices_to_param = [None]
            param_indices_from_rtnt = [None]

            # for bwd, since we already have a mapping between device param and device retent, 
            # we just need to add a mapping from device grad to host grad
            host_indices_from_grad = [None]
            grad_indices_to_host = [None]

            torch.cuda.nvtx.range_push("mask calculation loop")
            for i in range(1, bsz):
                curr_cam = ordered_cams[i]
                prev_cam = ordered_cams[i-1]
                curr_idx = retent_index[curr_cam]
                prev_idx = retent_index[prev_cam]

                idx_h = torch.nonzero(gs_bitmap[curr_cam] & ~gs_bitmap[prev_cam]).flatten()
                host_indices_to_param.append(idx_h)
                param_indices_from_host.append(torch.gather(curr_idx, dim=0, index=idx_h))

                idx_d = torch.nonzero(gs_bitmap[curr_cam] & gs_bitmap[prev_cam]).flatten() # overlap
                rtnt_indices_to_param.append(torch.gather(prev_idx, dim=0, index=idx_d))
                param_indices_from_rtnt.append(torch.gather(curr_idx, dim=0, index=idx_d))

                idx_g = torch.nonzero(gs_bitmap[prev_cam] & ~gs_bitmap[curr_cam]).flatten()
                host_indices_from_grad.append(idx_g)
                grad_indices_to_host.append(torch.gather(prev_idx, dim=0, index=idx_g))
            torch.cuda.nvtx.range_pop()

            rtnt_indices_from_grad = param_indices_from_rtnt
            grad_indices_to_rtnt = rtnt_indices_to_param

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            cat_update_ls_cpu = torch.empty_like(cat_update_ls, device="cpu", pin_memory=True)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls_cpu.copy_(cat_update_ls)
            update_ls_cpu = torch.split(cat_update_ls_cpu, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        elif order_calculation_version == 100:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("cumsum - 1")
            retent_index = torch.empty_like(gs_bitmap, dtype=torch.int64)
            for i in range(bsz):
                retent_index[i] = torch.cumsum(gs_bitmap[i], dim=0, dtype=torch.int64) - 1
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            # torch.flip(ordered_cams, dims=[0])
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            # calculate filters to index retent shs on gpu and shs to load from cpu
            host_indices_to_param = [None]
            param_indices_from_host = [None]
            rtnt_indices_to_param = [None]
            param_indices_from_rtnt = [None]

            # for bwd, since we already have a mapping between device param and device retent, 
            # we just need to add a mapping from device grad to host grad
            host_indices_from_grad = [None]
            grad_indices_to_host = [None]

            torch.cuda.nvtx.range_push("mask calculation loop")
            for i in range(1, bsz):
                curr_cam = ordered_cams[i]
                prev_cam = ordered_cams[i-1]
                curr_idx = retent_index[curr_cam]
                prev_idx = retent_index[prev_cam]

                idx_h = torch.nonzero(gs_bitmap[curr_cam] & ~gs_bitmap[prev_cam]).flatten()
                host_indices_to_param.append(idx_h)
                param_indices_from_host.append(torch.gather(curr_idx, dim=0, index=idx_h))

                idx_d = torch.nonzero(gs_bitmap[curr_cam] & gs_bitmap[prev_cam]).flatten() # overlap
                rtnt_indices_to_param.append(torch.gather(prev_idx, dim=0, index=idx_d))
                param_indices_from_rtnt.append(torch.gather(curr_idx, dim=0, index=idx_d))

                idx_g = torch.nonzero(gs_bitmap[prev_cam] & ~gs_bitmap[curr_cam]).flatten()
                host_indices_from_grad.append(idx_g)
                grad_indices_to_host.append(torch.gather(prev_idx, dim=0, index=idx_g))
            torch.cuda.nvtx.range_pop()

            rtnt_indices_from_grad = param_indices_from_rtnt
            grad_indices_to_rtnt = rtnt_indices_to_param

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            cat_update_ls_cpu = torch.empty_like(cat_update_ls, device="cpu", pin_memory=True)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls_cpu.copy_(cat_update_ls)
            update_ls_cpu = torch.split(cat_update_ls_cpu, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
        
    if args.overlap_cpuadam:
        # Define python thread for computing cpuadam
        def cpuadam_thread_v0(bsz,
                              n_gaussians,
                              microbatch_gradient_send_back_events,
                              thread_sync_signal_events,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")
            parameters.grad = parameters_grad

            cpu_adam.sparse_adam_inc_step() # this is related to lr. 
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.sparse_step(sparse_indices=finish_indices_filters[0], version=2, scale=1.0/bsz)
                torch.cuda.nvtx.range_pop()

            for i in range(0, bsz):
                torch.cuda.nvtx.range_push(f"cpuadam microbatch: {i} - [{iteration},{iteration+bsz})")
                
                thread_sync_signal_events[i].wait() # wait for the signal of finishing the i-th microbatch.

                finish_event = microbatch_gradient_send_back_events[i] # event of finishing i-th micro batch.
                finish_event.synchronize() # synchronize with the gpu event on computation stream.

                finish_indices_filter = finish_indices_filters[i+1] # torch int32 array on cpu
                if not args.stop_update_param and finish_indices_filter.shape[0] > 0: # the finish filter should not be empty
                    cpu_adam.sparse_step(sparse_indices=finish_indices_filter, version=2, scale=1.0/bsz)
                
                torch.cuda.nvtx.range_pop()

            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
            torch.cuda.nvtx.range_pop()

        def cpuadam_thread_v1(bsz,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            parameters.grad = parameters_grad / bsz
            cpu_adam.step()
            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 

        def cpuadam_thread_v2(bsz,
                           finish_indices_filters,
                           cpu_adam,
                           parameters,
                           parameters_grad):
            # parameters.grad = parameters_grad / bsz
            parameters.grad = parameters_grad
            cpu_adam.sparse_adam_inc_step()
            for filter in finish_indices_filters:
                cpu_adam.sparse_step(sparse_indices=filter, version=1, scale=1.0/bsz)
            parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 

        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        def cpuadam_thread_v4(bsz,
                              microbatch_gradient_send_back_events,
                              thread_sync_signal_events,
                              signal_tensor_pinned):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")
            for i in range(0, bsz):
                torch.cuda.nvtx.range_push(f"cpuadam microbatch: {i} - [{iteration},{iteration+bsz})")
                
                thread_sync_signal_events[i].wait() # wait for the signal of finishing the i-th microbatch.

                finish_event = microbatch_gradient_send_back_events[i] # event of finishing i-th micro batch.
                finish_event.synchronize() # synchronize with the gpu event on computation stream.

                signal_tensor_pinned[i] = 1 # set signal to 1 to notify gradients have been sent back to cpu

                torch.cuda.nvtx.range_pop()

        # Create thread for cpuadam
        if overlap_cpuadam_version == 0:
            thread_sync_signal_events = [threading.Event() for _ in range(bsz)]
            main_thread_sync_signal_idx = 0
            microbatch_gradient_send_back_events = [
                torch.cuda.Event() for _ in range(bsz)
            ]
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v0, args=(bsz,
                                                                        n_gaussians,
                                                                        microbatch_gradient_send_back_events,
                                                                        thread_sync_signal_events,
                                                                        finish_indices_filters,
                                                                        gaussians.optimizer.cpu_adam,
                                                                        gaussians._parameters,
                                                                        parameters_grad_buffer[:n_gaussians, :],
                                                                        ))
            cpuadam_worker.start()
        elif overlap_cpuadam_version == 1:
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v1, args=(bsz,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
        elif overlap_cpuadam_version == 2:
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v2, args=(bsz,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
        elif overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        elif overlap_cpuadam_version == 4:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
            
            thread_sync_signal_events = [threading.Event() for _ in range(bsz)]
            main_thread_sync_signal_idx = 0
            microbatch_gradient_send_back_events = [
                torch.cuda.Event() for _ in range(bsz)
            ]
            cpuadam_synchronization_worer = threading.Thread(target=cpuadam_thread_v4, args=(bsz,
                                                                                             microbatch_gradient_send_back_events,
                                                                                             thread_sync_signal_events,
                                                                                             signal_tensor_pinned))
            cpuadam_synchronization_worer.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            

    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    opacity_gpu = opacity_gpu_origin.detach().requires_grad_()
    scaling_gpu = scaling_gpu_origin.detach().requires_grad_()
    rotation_gpu = rotation_gpu_origin.detach().requires_grad_()

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retent = None
    shs_grad = None
    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream):
                # Forward pass
                shs = torch.empty(filters[micro_idx].shape[0], 48, device="cuda", requires_grad=True)

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retent = shs.detach()
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            shs_retent = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event

        with torch.cuda.stream(comm_stream):
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

                # Copy from retent
                send_shs2gpu_stream_retention(
                    shs_next,
                    gaussians._parameters,
                    shs_retent,
                    host_indices_to_param[micro_idx + 1],
                    rtnt_indices_to_param[micro_idx + 1],
                    param_indices_from_host[micro_idx + 1],
                    param_indices_from_rtnt[micro_idx + 1],
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        with torch.cuda.stream(comm_stream):
            last_microbatch_shs_grad = shs_grad
            # shs_grad = torch.empty_like(shs)
            shs_grad = torch.zeros_like(shs)
            shs_grad_init_event = torch.cuda.Event()
            shs_grad_init_event.record(comm_stream)


        if args.offload_shs_grad_before_every_microbatch and micro_idx > 0:
            with torch.cuda.stream(comm_stream):
                gpu2cpu_event.wait(comm_stream)
                # sync event of default_stream with comm_stream
                # timers.start("fused grad transfer") # rewrite the timer function.

                send_shs2cpu_grad_buffer_stream_retention(
                    last_microbatch_shs_grad,
                    parameters_grad_buffer[:N, :],
                    shs_grad,
                    host_indices_from_grad[micro_idx],
                    rtnt_indices_from_grad[micro_idx],
                    grad_indices_to_host[micro_idx],
                    grad_indices_to_rtnt[micro_idx],
                    True,
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                
                if args.overlap_cpuadam:
                    if args.overlap_cpuadam_version in [0, 4]:
                        event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                        event.record()
                        thread_sync_signal_events[main_thread_sync_signal_idx].set()
                        main_thread_sync_signal_idx += 1
                    elif args.overlap_cpuadam_version == 3:
                        # set signal to pinned memory to notify gradients have been sent back to cpu
                        diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                        microbatch_idx += 1

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
        shs_grad_init_event.wait(default_stream) # wait for `shs_grad` to finish init`
        # shs_grad.copy_(shs.grad)
        shs_grad.add_(shs.grad)

        ## free shs
        # shs.grad = None
        # del shs

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
                    if args.overlap_cpuadam_version in [0, 4]:
                        event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                        event.record()
                        thread_sync_signal_events[main_thread_sync_signal_idx].set()
                        main_thread_sync_signal_idx += 1
                    elif args.overlap_cpuadam_version == 3:
                        # set signal to pinned memory to notify gradients have been sent back to cpu
                        diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                        microbatch_idx += 1

                # timers.stop("fused grad transfer")

                # TODO: Free grads on gpu, I am not sure whether this is safe or not. 
                # We need to double check this. 
                # shs.grad = None

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
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
                if args.overlap_cpuadam_version in [0, 4]:
                    event = microbatch_gradient_send_back_events[main_thread_sync_signal_idx]
                    event.record()
                    thread_sync_signal_events[main_thread_sync_signal_idx].set()
                    main_thread_sync_signal_idx += 1
                elif args.overlap_cpuadam_version == 3:
                    # set signal to pinned memory to notify gradients have been sent back to cpu
                    diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        if args.overlap_cpuadam_version in [0, 4]:
            assert main_thread_sync_signal_idx == bsz, "main_thread_sync_signal_idx should be equal to bsz."
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
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

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses

def pipeline_offload_retention_optimized_impl(
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
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.offload_shs_grad_before_every_microbatch, "retention 2 currently requires disable offload_shs_grad_before_every_microbatc"

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    torch.cuda.nvtx.range_push("calculate_filters")
    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)
    torch.cuda.nvtx.range_pop()

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        if order_calculation_version == 2:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            retent_bitmap = gs_bitmap.clone()
            # the following cumsum is slow (14 ms on 10m rubble)
            retent_index = torch.cumsum(retent_bitmap, dim=1, dtype=torch.int64) - 1
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = [cur_cam]
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-1]
                gs_bitmap[:, filters[cur_cam]] = 0
                s_vec = torch.sum(gs_bitmap, dim=1, dtype=torch.int32)
                s_vec[ordered_cams] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(s_vec).item()
                ordered_cams.append(next_cam)
                next_update = torch.nonzero(gs_bitmap[next_cam, :]).flatten()
                update_ls.append(next_update)
            update_ls.append(not_touched_ids)
            update_ls.reverse()
            ordered_cams.reverse()
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            # calculate filters to index retent shs on gpu and shs to load from cpu
            host_indices_to_param = [None]
            param_indices_from_host = [None]
            rtnt_indices_to_param = [None]
            param_indices_from_rtnt = [None]

            # for bwd, since we already have a mapping between device param and device retent, 
            # we just need to add a mapping from device grad to host grad
            host_indices_from_grad = [None]
            grad_indices_to_host = [None]

            torch.cuda.nvtx.range_push("mask calculation loop")
            for i in range(1, bsz):
                curr_cam = ordered_cams[i]
                prev_cam = ordered_cams[i-1]
                curr_idx = retent_index[curr_cam]
                prev_idx = retent_index[prev_cam]

                idx_h = torch.nonzero(retent_bitmap[curr_cam] & ~retent_bitmap[prev_cam]).flatten()
                host_indices_to_param.append(idx_h)
                param_indices_from_host.append(torch.gather(curr_idx, dim=0, index=idx_h))

                idx_d = torch.nonzero(retent_bitmap[curr_cam] & retent_bitmap[prev_cam]).flatten() # overlap
                rtnt_indices_to_param.append(torch.gather(prev_idx, dim=0, index=idx_d))
                param_indices_from_rtnt.append(torch.gather(curr_idx, dim=0, index=idx_d))

                idx_g = torch.nonzero(retent_bitmap[prev_cam] & ~retent_bitmap[curr_cam]).flatten()
                host_indices_from_grad.append(idx_g)
                grad_indices_to_host.append(torch.gather(prev_idx, dim=0, index=idx_g))
            torch.cuda.nvtx.range_pop()

            rtnt_indices_from_grad = param_indices_from_rtnt
            grad_indices_to_rtnt = rtnt_indices_to_param

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls = cat_update_ls.cpu()
            update_ls_cpu = torch.split(cat_update_ls, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians

        elif order_calculation_version == 3:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("cumsum - 1")
            retent_index = torch.empty_like(gs_bitmap, dtype=torch.int64)
            for i in range(bsz):
                retent_index[i] = torch.cumsum(gs_bitmap[i], dim=0, dtype=torch.int64) - 1
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[0] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[i - 1]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[:i].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[i] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.flip(ordered_cams, dims=[0])
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            # calculate filters to index retent shs on gpu and shs to load from cpu
            host_indices_to_param = [None]
            param_indices_from_host = [None]
            rtnt_indices_to_param = [None]
            param_indices_from_rtnt = [None]

            # for bwd, since we already have a mapping between device param and device retent, 
            # we just need to add a mapping from device grad to host grad
            host_indices_from_grad = [None]
            grad_indices_to_host = [None]

            torch.cuda.nvtx.range_push("mask calculation loop")
            for i in range(1, bsz):
                curr_cam = ordered_cams[i]
                prev_cam = ordered_cams[i-1]
                curr_idx = retent_index[curr_cam]
                prev_idx = retent_index[prev_cam]

                idx_h = torch.nonzero(gs_bitmap[curr_cam] & ~gs_bitmap[prev_cam]).flatten()
                host_indices_to_param.append(idx_h)
                param_indices_from_host.append(torch.gather(curr_idx, dim=0, index=idx_h))

                idx_d = torch.nonzero(gs_bitmap[curr_cam] & gs_bitmap[prev_cam]).flatten() # overlap
                rtnt_indices_to_param.append(torch.gather(prev_idx, dim=0, index=idx_d))
                param_indices_from_rtnt.append(torch.gather(curr_idx, dim=0, index=idx_d))

                idx_g = torch.nonzero(gs_bitmap[prev_cam] & ~gs_bitmap[curr_cam]).flatten()
                host_indices_from_grad.append(idx_g)
                grad_indices_to_host.append(torch.gather(prev_idx, dim=0, index=idx_g))
            torch.cuda.nvtx.range_pop()

            rtnt_indices_from_grad = param_indices_from_rtnt
            grad_indices_to_rtnt = rtnt_indices_to_param

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            cat_update_ls_cpu = torch.empty_like(cat_update_ls, device="cpu", pin_memory=True)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls_cpu.copy_(cat_update_ls)
            update_ls_cpu = torch.split(cat_update_ls_cpu, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        # This version only calculates the order and cpuadam update_ls based on v3
        #TODO: downsample this
        elif order_calculation_version == 4:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            torch.cuda.nvtx.range_push("transfer cpuadam update list to cpu")
            cat_update_ls = torch.cat(update_ls, dim=0).to(torch.int32)
            # cat_update_ls = torch.cat(update_ls, dim=0)
            cat_update_ls_cpu = torch.empty_like(cat_update_ls, device="cpu", pin_memory=True)
            update_ls_dim = [len(update) for update in update_ls]
            cat_update_ls_cpu.copy_(cat_update_ls)
            update_ls_cpu = torch.split(cat_update_ls_cpu, update_ls_dim, dim=0)
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
        
    if args.overlap_cpuadam:
        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()
            pass

        # Create thread for cpuadam
        if overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
    
    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    opacity_gpu = opacity_gpu_origin.detach().requires_grad_()
    scaling_gpu = scaling_gpu_origin.detach().requires_grad_()
    rotation_gpu = rotation_gpu_origin.detach().requires_grad_()

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retent = None
    shs_grad = None
    with torch.cuda.stream(comm_stream):
        retention_vec = torch.full((N,), -1, dtype=torch.int64, device="cuda")
        shs_grad_next = torch.zeros((len(filters[0]), 48), device="cuda")

    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream):
                # Forward pass
                shs = torch.empty(this_filter_len, 48, device="cuda", requires_grad=True)

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retent = shs.detach()
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            shs_retent = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event

        with torch.cuda.stream(comm_stream):
            #TODO: Can we do this inside comm kernels? Might be slow.
            retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.arange(filters[micro_idx].shape[0], dtype=torch.int64, device="cuda"))
            
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")
                
                send_shs2gpu_stream_retention2_64(
                    shs_next, # shs to fill
                    gaussians._parameters, # shs on host
                    shs_retent, # shs from last iter
                    filters[micro_idx+1],
                    retention_vec, # with info of last iter index
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        with torch.cuda.stream(comm_stream):
            shs_grad = shs_grad_next
            shs_grad_init_event = torch.cuda.Event()
            shs_grad_init_event.record(comm_stream)

        torch.cuda.nvtx.range_push("forward_pass")
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
        shs_grad_init_event.wait(default_stream) # wait for `shs_grad` to finish init`
        shs_grad.add_(shs.grad)

        # free shs
        shs.grad = None
        del shs

        losses.append(loss.detach())

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        if not args.offload_shs_grad_before_every_microbatch:
            if micro_idx < num_micro_batches - 1:
                with torch.cuda.stream(comm_stream):
                    shs_grad_next = torch.zeros_like(shs_next, device="cuda")

                    gpu2cpu_event.wait(comm_stream)
                    # sync event of default_stream with comm_stream

                    send_shs2cpu_grad_buffer_stream_retention2_64(
                        shs_grad, # current iteration grad
                        parameters_grad_buffer[:N], # shs grad on host
                        shs_grad_next, # shs grad retention for next iteration
                        filters[micro_idx],
                        filters[micro_idx+1],
                        retention_vec,
                        True,
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D
                    )

                    # reset retention_vec
                    retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.full((this_filter_len,), -1, dtype=torch.int64, device="cuda"))

                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1
                    
            else:
                with torch.cuda.stream(comm_stream):
                    gpu2cpu_event.wait(comm_stream)

                    send_shs2cpu_grad_buffer_stream(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        filters[-1],
                        True,
                        16, 256,
                    )
                    
                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        del batched_means2D, batched_radiis

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
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

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses

def pipeline_offload_retention_optimized_v3_impl(
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
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.offload_shs_grad_before_every_microbatch, "retention 2 currently requires disable offload_shs_grad_before_every_microbatc"

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    torch.cuda.nvtx.range_push("calculate_filters")
    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)
    torch.cuda.nvtx.range_pop()

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        # This version only calculates the order and cpuadam update_ls based on v3
        #TODO: downsample this
        if order_calculation_version == 4:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            cnt_h = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]

            torch.cuda.nvtx.range_push("precompute sums")
            for i in range(bsz-1):
                this_bit = gs_bitmap[ordered_cams[i]]
                next_bit = gs_bitmap[ordered_cams[i+1]]
                cnt_h[i] = torch.sum(~this_bit & next_bit)
                cnt_d[i] = torch.sum(this_bit & next_bit)
                cnt_g[i] = torch.sum(this_bit & ~next_bit)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]

            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu
            del gs_bitmap

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
        
    if args.overlap_cpuadam:
        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()
            pass

        # Create thread for cpuadam
        if overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
    
    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    opacity_gpu = opacity_gpu_origin.detach().requires_grad_()
    scaling_gpu = scaling_gpu_origin.detach().requires_grad_()
    rotation_gpu = rotation_gpu_origin.detach().requires_grad_()

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retents = [torch.empty(0) for i in range(num_micro_batches)]
    shs_grad = None
    with torch.cuda.stream(comm_stream):
        this_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        next_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        retention_vec = torch.empty((N,), dtype=torch.int32, device="cuda")

    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            shs = torch.empty(this_filter_len, 48, device="cuda", requires_grad=True)

            with torch.cuda.stream(comm_stream):
                # Forward pass
                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retents[micro_idx] = shs.detach()
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
            
        else:
            shs = shs_next
            shs_retents[micro_idx] = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event

        if micro_idx < num_micro_batches - 1:
            # Before allocating memory for next shs, free previous retent.
            if micro_idx > 0:
                next_cpu2gpu_event.wait(default_stream) # Make sure cpu2gpu comm in prev iter has completed and retent is no longer needed.
                shs_retents[micro_idx-1] = None
            shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

            with torch.cuda.stream(comm_stream):
                if micro_idx == 0:
                    this_bit.scatter_(dim=0, index=filters[micro_idx], src=torch.ones(filters[micro_idx].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                else:
                    this_bit, next_bit = next_bit, this_bit
                    next_bit.scatter_(dim=0, index=filters[micro_idx-1], src=torch.zeros(filters[micro_idx-1].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                
                # NOTE: Here we use `torch.nonzero_static`(torch2.6) instead of `torch.nonzero` to avoid h2d sync.
                # This stems from the need to know #nonzero_elem before cuda kernel launch.
                # When using `torch.nonzero_static`, `size`` need to be a scalar on host, otherwise it falls back to blocking.
                # retention_vec: next index
                retention_vec.scatter_(dim=0, index=filters[micro_idx+1], src=torch.arange(filters[micro_idx+1].shape[0], dtype=torch.int32, device="cuda"))
                bit_h = ~this_bit & next_bit
                idx_h = torch.empty((cnt_h[micro_idx],), dtype=torch.int64, device="cuda")
                idx_h = torch.nonzero_static(bit_h, size=cnt_h[micro_idx]).flatten()
                host_indices_to_param = idx_h.to(torch.int32)
                param_indices_from_host = torch.gather(retention_vec, dim=0, index=idx_h)
                del idx_h, bit_h
                
                bit_d = this_bit & next_bit
                idx_d = torch.nonzero_static(bit_d, size=cnt_d[micro_idx]).flatten()
                param_indices_from_rtnt = torch.gather(retention_vec, dim=0, index=idx_d)
                del bit_d

                retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.arange(filters[micro_idx].shape[0], dtype=torch.int32, device="cuda"))
                rtnt_indices_to_param = torch.gather(retention_vec, dim=0, index=idx_d)
                del idx_d
                
                send_shs2gpu_stream_retention(
                    shs_next, # shs to fill
                    gaussians._parameters, # shs on host
                    shs_retents[micro_idx], # shs from last iter
                    host_indices_to_param,
                    rtnt_indices_to_param,
                    param_indices_from_host,
                    param_indices_from_rtnt,
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        torch.cuda.nvtx.range_push("forward_pass")
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

        # Before allocating memory for this grad, free previous grad.
        if micro_idx > 0:
            next_gpu2cpu_event.wait(default_stream) # Make sure gpu2cpu comm in prev iter has completed and prev grad is no longer needed.
            del shs_grad

        torch.cuda.nvtx.range_push("backward_pass")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        del rendered_image

        if micro_idx == 0:
            shs_grad = torch.empty_like(shs, device="cuda")
            shs_grad.copy_(shs.grad)
        else:
            shs_grad = shs_grad_next
            shs_grad.add_(shs.grad)

        # Free shs.
        shs.grad = None
        del shs, filtered_shs

        losses.append(loss.detach())
        del loss

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        if not args.offload_shs_grad_before_every_microbatch:
            if micro_idx < num_micro_batches - 1:
                shs_grad_next = torch.zeros_like(shs_next, device="cuda")
                shs_grad_next_init_event = torch.cuda.Event(enable_timing=True)
                shs_grad_next_init_event.record(default_stream)

                with torch.cuda.stream(comm_stream):
                    rtnt_indices_from_grad = param_indices_from_rtnt
                    grad_indices_to_rtnt = rtnt_indices_to_param

                    bit_g = this_bit & ~next_bit
                    idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()

                    host_indices_from_grad = idx_g.to(torch.int32)
                    grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
                    del idx_g, bit_g

                    gpu2cpu_event.wait(comm_stream)
                    shs_grad_next_init_event.wait(comm_stream) # Wait for all preceeding init to finish
                    # sync event of default_stream with comm_stream

                    send_shs2cpu_grad_buffer_stream_retention(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        shs_grad_next,
                        host_indices_from_grad,
                        rtnt_indices_from_grad,
                        grad_indices_to_host,
                        grad_indices_to_rtnt,
                        True,
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D
                    )
                    next_gpu2cpu_event = torch.cuda.Event(enable_timing=True)
                    next_gpu2cpu_event.record(comm_stream)

                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1
                    
            else:
                with torch.cuda.stream(comm_stream):
                    gpu2cpu_event.wait(comm_stream)

                    send_shs2cpu_grad_buffer_stream(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        filters[-1],
                        True,
                        grid_size, block_size
                    )
                    
                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        del batched_means2D, batched_radiis

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
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

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses

def pipeline_offload_retention_optimized_v4_impl(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    log_this_batch_filter=False,
):
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.offload_shs_grad_before_every_microbatch, "retention 4 currently requires disable offload_shs_grad_before_every_microbatc"

    # prepare all parameters
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation
    bsz = len(batched_cameras)
    n_gaussians = xyz_gpu.shape[0]

    torch.cuda.nvtx.range_push("calculate_filters")
    # calculate gaussian visible filters for all cameras
    filters, camera_ids, gaussian_ids = calculate_filters(
        batched_cameras,
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    ) # list of GPU long tensors. len(cameras)
    torch.cuda.nvtx.range_pop()

    if log_this_batch_filter:
        f_dump = {
            "iteration": iteration,
            "filters": [f.tolist() for f in filters],
        }
        with open(os.path.join(args.model_path, "sampled_filters.log"), 'a') as file:
            file.write(json.dumps(f_dump) + "\n")

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        # This version only calculates the order and cpuadam update_ls based on v3
        #TODO: downsample this
        if order_calculation_version == 4:
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("init lists")
            not_touched_ids = torch.nonzero(torch.all(gs_bitmap == 0, dim=0)).flatten()
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            cnt_h = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]
            sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

            torch.cuda.nvtx.range_push("precompute sums")
            for i in range(bsz-1):
                this_bit = gs_bitmap[ordered_cams[i]]
                next_bit = gs_bitmap[ordered_cams[i+1]]
                cnt_h[i] = torch.sum(~this_bit & next_bit)
                cnt_d[i] = torch.sum(this_bit & next_bit)
                cnt_g[i] = torch.sum(this_bit & ~next_bit)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]

            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            del gs_bitmap

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
        
    if args.overlap_cpuadam:
        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        # Create thread for cpuadam
        if overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
    
    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    opacity_gpu = opacity_gpu_origin.detach().requires_grad_()
    scaling_gpu = scaling_gpu_origin.detach().requires_grad_()
    rotation_gpu = rotation_gpu_origin.detach().requires_grad_()

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retents = [None for i in range(num_micro_batches)]

    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    with torch.cuda.stream(comm_stream), torch.no_grad():
        this_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        next_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        retention_vec = torch.empty((N,), dtype=torch.int32, device="cuda")

        shs_grad = torch.zeros(filters[0].shape[0], 48, device="cuda")
        shs_grad_init_event = torch.cuda.Event()
        shs_grad_init_event.record(comm_stream)

    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream), torch.no_grad():
                # Forward pass
                shs = torch.empty(this_filter_len, 48, device="cuda", requires_grad=True)

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retents[micro_idx] = shs.detach()
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
            
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            shs_retents[micro_idx] = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event


        with torch.cuda.stream(comm_stream), torch.no_grad():
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

                # compute indices on the fly
                if micro_idx == 0:
                    this_bit.scatter_(dim=0, index=filters[micro_idx], src=torch.ones(filters[micro_idx].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                else:
                    this_bit, next_bit = next_bit, this_bit
                    next_bit.scatter_(dim=0, index=filters[micro_idx-1], src=torch.zeros(filters[micro_idx-1].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                
                # NOTE: Here we use `torch.nonzero_static`(torch2.6) instead of `torch.nonzero` to avoid h2d sync.
                # This stems from the need to know #nonzero_elem before cuda kernel launch.
                # When using `torch.nonzero_static`, `size`` need to be a scalar on host, otherwise it falls back to blocking.
                # retention_vec: next index
                retention_vec.scatter_(dim=0, index=filters[micro_idx+1], src=torch.arange(filters[micro_idx+1].shape[0], dtype=torch.int32, device="cuda"))
                # idx_h = torch.nonzero(~this_bit & next_bit).flatten() # torch.nonzero() blocks cpu!!!
                bit_h = ~this_bit & next_bit
                idx_h = torch.empty((cnt_h[micro_idx],), dtype=torch.int64, device="cuda")
                idx_h = torch.nonzero_static(bit_h, size=cnt_h[micro_idx]).flatten()
                host_indices_to_param = idx_h.to(torch.int32)
                param_indices_from_host = torch.gather(retention_vec, dim=0, index=idx_h)
                del idx_h, bit_h
                
                # idx_d = torch.nonzero(this_bit & next_bit).flatten() # overlap # torch.nonzero() blocks cpu!!!
                bit_d = this_bit & next_bit
                idx_d = torch.nonzero_static(bit_d, size=cnt_d[micro_idx]).flatten()
                param_indices_from_rtnt = torch.gather(retention_vec, dim=0, index=idx_d) # reused in gpu2cpu comm
                del bit_d

                # retention_vec: this index
                retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.arange(filters[micro_idx].shape[0], dtype=torch.int32, device="cuda"))
                rtnt_indices_to_param = torch.gather(retention_vec, dim=0, index=idx_d) # reused in gpu2cpu comm
                del idx_d
                
                send_shs2gpu_stream_retention(
                    shs_next, # shs to fill
                    gaussians._parameters, # shs on host
                    shs_retents[micro_idx], # shs from last iter
                    host_indices_to_param,
                    rtnt_indices_to_param,
                    param_indices_from_host,
                    param_indices_from_rtnt,
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                del host_indices_to_param, param_indices_from_host
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        torch.cuda.nvtx.range_push("forward_pass")
        filtered_xyz_gpu = torch.gather(xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3))
        filtered_opacity_gpu = torch.gather(opacity_gpu, 0, this_filter.reshape(-1, 1))
        filtered_scaling_gpu = torch.gather(scaling_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3))
        filtered_rotation_gpu = torch.gather(rotation_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 4))
        # sync event of comm_stream with default_stream to make sure shs has been loaded to gpu
        cpu2gpu_event.wait(default_stream)
        filtered_shs = shs.requires_grad_(False) # this is a view of the original shs.
        # filtered_shs = shs

        filtered_filtered_xyz_gpu = filtered_xyz_gpu.detach().requires_grad_()

        # preprocess
        rendered_image, batched_means2D, batched_radiis, batched_colors_detached, dirs  = pipeline_forward_one_step_shs_inplace(filtered_opacity_gpu,
                                                filtered_scaling_gpu,
                                                filtered_rotation_gpu,
                                                # filtered_xyz_gpu,
                                                filtered_filtered_xyz_gpu,
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
        
        shs_grad_init_event.wait(default_stream) # wait for `shs_grad` to finish init`

        v_dirs = spherical_harmonics_bwd_inplace(degrees_to_use=gaussians.active_sh_degree, dirs=dirs, coeffs=filtered_shs.reshape(1, -1, 16, 3), v_coeffs=shs_grad, v_colors=batched_colors_detached.grad)
        dirs.backward(v_dirs)
        filtered_xyz_gpu.backward(filtered_filtered_xyz_gpu.grad)
        torch.cuda.nvtx.range_pop()

        del rendered_image, batched_colors_detached, dirs, v_dirs, filtered_filtered_xyz_gpu

        # free shs
        shs = None
        del filtered_xyz_gpu, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu, filtered_shs

        losses.append(loss.detach())
        del loss

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        if not args.offload_shs_grad_before_every_microbatch:
            if micro_idx < num_micro_batches - 1:
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    # compute indices on the fly

                    rtnt_indices_from_grad = param_indices_from_rtnt
                    grad_indices_to_rtnt = rtnt_indices_to_param

                    # idx_g = torch.nonzero(this_bit & ~next_bit).flatten() # torch.nonzero() blocks cpu!!!
                    bit_g = this_bit & ~next_bit
                    idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()

                    host_indices_from_grad = idx_g.to(torch.int32)
                    grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
                    del idx_g, bit_g

                    # sync event of default_stream with comm_stream
                    gpu2cpu_event.wait(comm_stream)
                    shs_retents[micro_idx] = None
                    shs_grad_next = torch.zeros_like(shs_next, device="cuda")

                    send_shs2cpu_grad_buffer_stream_retention(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        shs_grad_next,
                        host_indices_from_grad,
                        rtnt_indices_from_grad,
                        grad_indices_to_host,
                        grad_indices_to_rtnt,
                        True,
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D
                    )
                    # del shs_grad
                    shs_grad = shs_grad_next
                    # del host_indices_from_grad, rtnt_indices_from_grad, grad_indices_to_host, grad_indices_to_rtnt, param_indices_from_rtnt, rtnt_indices_to_param
                    shs_grad_init_event.record(comm_stream)

                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1
                    
            else:
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    gpu2cpu_event.wait(comm_stream)

                    send_shs2cpu_grad_buffer_stream(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        filters[-1],
                        True,
                        grid_size, block_size
                    )
                    
                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

    opacity_gpu_origin.backward(opacity_gpu.grad)
    scaling_gpu_origin.backward(scaling_gpu.grad)
    rotation_gpu_origin.backward(rotation_gpu.grad)

    if args.overlap_cpuadam:
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
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

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses, ordered_cams, sparsity

# v5: A memory efficient version based on v4.
def pipeline_offload_retention_optimized_v5_impl(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    log_this_batch_filter=False,
):
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.offload_shs_grad_before_every_microbatch, "retention 4 currently requires disable offload_shs_grad_before_every_microbatc"

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    with torch.no_grad():
        # prepare all parameters
        xyz_gpu = gaussians.get_xyz
        opacity_gpu_origin = gaussians.get_opacity
        scaling_gpu_origin = gaussians.get_scaling
        rotation_gpu_origin = gaussians.get_rotation

        torch.cuda.nvtx.range_push("calculate_filters")
        # calculate gaussian visible filters for all cameras
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        ) # list of GPU long tensors. len(cameras)
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()

    if log_this_batch_filter:
        f_dump = {
            "iteration": iteration,
            "filters": [f.tolist() for f in filters],
        }
        with open(os.path.join(args.model_path, "sampled_filters.log"), 'a') as file:
            file.write(json.dumps(f_dump) + "\n")

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        if order_calculation_version == -1:
            # Computes cpu-adam update list using random order.
            ordered_cams = range(bsz)
            sparsity = None

            bool_filters = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            bool_filters[camera_ids, gaussian_ids] = 1
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

            torch.cuda.nvtx.range_push("precompute sums")
            cnt_h = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            for i in range(bsz-1):
                this_bit = bool_filters[ordered_cams[i]]
                next_bit = bool_filters[ordered_cams[i+1]]
                cnt_h[i] = torch.sum(~this_bit & next_bit)
                cnt_d[i] = torch.sum(this_bit & next_bit)
                cnt_g[i] = torch.sum(this_bit & ~next_bit)
            torch.cuda.nvtx.range_pop()
            del bool_filters


            last_calc_percamera_counts_cpu = last_calc_percamera_counts.cpu().tolist()
            last_calc_percamera_counts_cpu = [not_touched_gaussian_ids.shape[0]] + last_calc_percamera_counts_cpu
            last_calc_gaussianids = torch.cat([not_touched_gaussian_ids, last_calc_gaussianids] + [cnt_h, cnt_d, cnt_g], dim=0).to(torch.int32)
            last_calc_gaussianids_cpu = torch.empty_like(last_calc_gaussianids, device="cpu", pin_memory=True)
            last_calc_gaussianids_cpu.copy_(last_calc_gaussianids)
            last_calc_gaussian_ids_per_camera = torch.split(last_calc_gaussianids_cpu, last_calc_percamera_counts_cpu + [bsz-1]*3)
            cnt_h = last_calc_gaussian_ids_per_camera[-3]
            cnt_d = last_calc_gaussian_ids_per_camera[-2]
            cnt_g = last_calc_gaussian_ids_per_camera[-1]
            last_calc_gaussian_ids_per_camera = last_calc_gaussian_ids_per_camera[:len(last_calc_gaussian_ids_per_camera)-3]

            assert sum(last_calc_percamera_counts_cpu) == n_gaussians, "sum(last_calc_percamera_counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
            finish_indices_filters = last_calc_gaussian_ids_per_camera

            if args.sparse_adam:
                src = torch.zeros((len(not_touched_gaussian_ids),), dtype=torch.bool, device="cuda")
                visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_gaussian_ids, src=src)
            else:
                visibility_mask = None
            

        elif order_calculation_version == 4:
        # This version only calculates the order and cpuadam update_ls based on v3
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("init lists")
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            cnt_h = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)
            
            col_to_reset = next_update
            zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
            not_touched_ids = torch.nonzero(zero_bitvec).flatten()

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()
            del not_touched_ids

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]
            sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

            torch.cuda.nvtx.range_push("precompute sums")
            for i in range(bsz-1):
                this_bit = gs_bitmap[ordered_cams[i]]
                next_bit = gs_bitmap[ordered_cams[i+1]]
                cnt_h[i] = torch.sum(~this_bit & next_bit)
                cnt_d[i] = torch.sum(this_bit & next_bit)
                cnt_g[i] = torch.sum(this_bit & ~next_bit)
            torch.cuda.nvtx.range_pop()
            del gs_bitmap

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]

            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        # v5: Optimized the bitmap for memory based on v4.
        # FIXME: Element-wise bit operations were very slow.
        elif order_calculation_version == 5:
            match bsz:
                case 4 | 8:
                    dtype = torch.int8
                case 16:
                    dtype = torch.int16
                case 32:
                    dtype = torch.int32
                case 64:
                    dtype = torch.int64
                case _:
                    raise ValueError("Currently supported bsz: (4, 8, 16, 32, 64).")

            torch.cuda.nvtx.range_push("init bitmap and vecs")
            gs_bitmap = torch.zeros((n_gaussians), dtype=dtype, device="cuda")
            for i, f in enumerate(filters):
                gs_bitmap.scatter_add_(dim=0, src=torch.ones((len(f),), dtype=dtype, device="cuda"), index=f)
                if i < bsz - 1:
                    gs_bitmap = gs_bitmap << 1
            
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("init lists")
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            # update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            update_ls = [torch.nonzero(gs_bitmap >> (bsz-1-cur_cam) & 1).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            tmp_vec = torch.empty((n_gaussians,), dtype=torch.uint8, device="cuda")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                if i > 1:
                    col_to_reset = next_update
                else:
                    diff_gaussian_rasterization._C.map_bit_and_vec(gs_bitmap, zero_bitvec, bsz-1-cur_cam, tmp_vec)
                    col_to_reset = next_update if i > 1 else torch.nonzero(tmp_vec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))

                reset_col_gathered = torch.gather(gs_bitmap, dim=0, index=col_to_reset)
                reset_bitmap = torch.empty((bsz, len(col_to_reset)), dtype=torch.uint8, device="cuda")
                diff_gaussian_rasterization._C.extract_reset_bitmap(reset_col_gathered, reset_bitmap)
                reset_sum = torch.sum(reset_bitmap, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam

                diff_gaussian_rasterization._C.map_bit_and_vec(gs_bitmap, zero_bitvec, bsz-1-next_cam, tmp_vec)
                next_update = torch.nonzero(tmp_vec).flatten()
                update_ls.append(next_update)
    
            col_to_reset = next_update
            zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
            not_touched_ids = torch.nonzero(zero_bitvec).flatten()
            if args.sparse_adam:
                src = torch.zeros((len(not_touched_ids),), dtype=torch.bool, device="cuda")
                visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_ids, src=src)
            else:
                visibility_mask = None

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()
            del not_touched_ids

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]
            sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

            torch.cuda.nvtx.range_push("precompute sums")
            cnt_h = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            hdg_vec = torch.empty((3, n_gaussians), dtype=torch.uint8, device="cuda")
            for i in range(bsz-1):
                diff_gaussian_rasterization._C.generate_hdg(gs_bitmap, bsz-1-ordered_cams[i], bsz-1-ordered_cams[i+1], hdg_vec)
                cnt_hdg = torch.sum(hdg_vec, dim=1)
                cnt_h[i] = cnt_hdg[0]
                cnt_d[i] = cnt_hdg[1]
                cnt_g[i] = cnt_hdg[2]

            torch.cuda.nvtx.range_pop()
            del gs_bitmap, hdg_vec

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians, f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"

        elif order_calculation_version == 6:
            match bsz:
                case 4 | 8:
                    dtype = torch.int8
                case 16:
                    dtype = torch.int16
                case 32:
                    dtype = torch.int32
                case 64:
                    dtype = torch.int64
                case _:
                    raise ValueError("Currently supported bsz: (4, 8, 16, 32, 64).")

            def order_cal_6(filters, batched_cameras):
                torch.cuda.nvtx.range_push("init bitmap and vecs")
                gs_bitmap = torch.zeros((n_gaussians), dtype=dtype, device="cuda")
                for i, f in enumerate(filters):
                    gs_bitmap.scatter_add_(dim=0, src=torch.ones((len(f),), dtype=dtype, device="cuda"), index=f)
                    if i < bsz - 1:
                        gs_bitmap = gs_bitmap << 1
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("generate distance matrix")
                # Downsample.
                n_sampled = n_gaussians // bsz**2
                sampled_gaussian_ids = torch.randperm(n_gaussians, generator=perm_generator, device="cuda")[:n_sampled]
                sampled_bitmap = torch.gather(input=gs_bitmap, dim=0, index=sampled_gaussian_ids)
                # Unzip the bimap.
                unziped = torch.empty((bsz, n_sampled), dtype=torch.uint8, device="cuda")
                for i in range(bsz):
                    unziped[i] = (sampled_bitmap & 1).to(torch.uint8)
                    sampled_bitmap >> 1
                # Compute distance matrix. FIXME: need a better way with less memory
                distance_matrix = (unziped.unsqueeze(1) ^ unziped.unsqueeze(0)).sum(dim=-1).tolist() # intermediate result: (bsz, bsz, n_sampled) = n_gaussians
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("solve order: tsp")
                ordered_cams = fast_tsp.find_tour(distance_matrix, 0.001)
                torch.cuda.nvtx.range_pop()
                batched_cameras = [batched_cameras[i] for i in ordered_cams]
                filters = [filters[i] for i in ordered_cams]
                sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

                torch.cuda.nvtx.range_push("generate cpuadam update ls")
                gs_bitmap.zero_()
                for i, f in enumerate(filters):
                    gs_bitmap.scatter_add_(dim=0, src=torch.ones((len(f),), dtype=dtype, device="cuda"), index=f)
                    if i < bsz - 1:
                        gs_bitmap = gs_bitmap << 1
                ffs = torch.empty(n_gaussians, dtype=torch.uint8, device="cuda")
                diff_gaussian_rasterization._C.extract_ffs(gs_bitmap, ffs)
                sorted_ffs, indices = torch.sort(ffs)
                elems, counts = torch.unique_consecutive(sorted_ffs, return_counts=True)
                update_ls = torch.split(indices, counts.tolist(), dim=0)
                update_ls = list(update_ls)
                for i in range(bsz + 1):
                    if i not in elems: # check if there is empty update
                        update_ls.insert(i, torch.tensor([], device="cuda"))
                update_ls = [update_ls[0]] + update_ls[:0:-1]

                if args.sparse_adam:
                    not_touched_ids = update_ls[0]
                    src = torch.zeros((len(not_touched_ids),), dtype=torch.bool, device="cuda")
                    visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_ids, src=src)
                else:
                    visibility_mask = None
                
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("precompute sums")
                cnt_h = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
                cnt_d = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
                cnt_g = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
                hdg_vec = torch.empty((3, n_gaussians), dtype=torch.uint8, device="cuda")
                for i in range(bsz-1):
                    diff_gaussian_rasterization._C.generate_hdg(gs_bitmap, bsz-1-i, bsz-1-i-1, hdg_vec)
                    cnt_hdg = torch.sum(hdg_vec, dim=1)
                    cnt_h[i] = cnt_hdg[0]
                    cnt_d[i] = cnt_hdg[1]
                    cnt_g[i] = cnt_hdg[2]

                torch.cuda.nvtx.range_pop()
                del gs_bitmap, hdg_vec

                torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
                data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
                cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
                cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
                data2cpu_dim = [len(d) for d in data2cpu_ls]
                cat_data2cpu_h.copy_(cat_data2cpu)
                data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
                assert len(data2cpu_ls_h) == bsz + 4
                update_ls_cpu = data2cpu_ls_h[:bsz+1]
                cnt_h = data2cpu_ls_h[-3]
                cnt_d = data2cpu_ls_h[-2]
                cnt_g = data2cpu_ls_h[-1]
                torch.cuda.nvtx.range_pop()

                finish_indices_filters = update_ls_cpu

                assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
                assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians, f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"

                return finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask

            finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask = order_cal_6(filters, batched_cameras)

        elif order_calculation_version == 7:
            match bsz:
                case 4 | 8:
                    dtype = torch.int8
                case 16:
                    dtype = torch.int16
                case 32:
                    dtype = torch.int32
                case 64:
                    dtype = torch.int64
                case _:
                    raise ValueError("Currently supported bsz: (4, 8, 16, 32, 64).")

            def order_cal_7(filters, batched_cameras):
                torch.cuda.nvtx.range_push("init bitmap and vecs")
                gs_bitmap = torch.zeros((n_gaussians), dtype=dtype, device="cuda")
                # Encode bitmap: MSB->first microbatch; LSB->last microbatch
                for i, f in enumerate(filters):
                    diff_gaussian_rasterization._C.scatter_to_bit(gs_bitmap, f, bsz-1-i)
    
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("generate distance matrix")
                # Downsample.
                if bsz >= 32:
                    n_sampled = n_gaussians // bsz**2
                else:
                    n_sampled = n_gaussians // 32
                sampled_gaussian_ids = torch.randperm(n_gaussians, generator=perm_generator, device="cuda")[:n_sampled]
                sampled_bitmap = torch.gather(input=gs_bitmap, dim=0, index=sampled_gaussian_ids)
                # Unzip the bimap.
                unziped = torch.empty((bsz, n_sampled), dtype=torch.uint8, device="cuda")
                for i in range(bsz):
                    unziped[i] = (sampled_bitmap & 1).to(torch.uint8)
                    sampled_bitmap >> 1
                # Compute distance matrix. FIXME: need a better way with less memory
                distance_matrix = (unziped.unsqueeze(1) ^ unziped.unsqueeze(0)).sum(dim=-1).tolist() # intermediate result: (bsz, bsz, n_sampled) = n_gaussians
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("solve order: tsp")
                ordered_cams = fast_tsp.find_tour(distance_matrix, 0.001)
                torch.cuda.nvtx.range_pop()
                batched_cameras = [batched_cameras[i] for i in ordered_cams]
                filters = [filters[i] for i in ordered_cams]
                sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

                torch.cuda.nvtx.range_push("generate cpuadam update ls")
                # Re-encode the bitmap based on given order
                gs_bitmap.zero_()
                for i, f in enumerate(filters):
                    diff_gaussian_rasterization._C.scatter_to_bit(gs_bitmap, f, bsz-1-i)

                ffs = torch.empty(n_gaussians, dtype=torch.uint8, device="cuda")
                diff_gaussian_rasterization._C.extract_ffs(gs_bitmap, ffs)
                sorted_ffs, indices = torch.sort(ffs)
                elems, counts = torch.unique_consecutive(sorted_ffs, return_counts=True)
                update_ls = torch.split(indices, counts.tolist(), dim=0)
                update_ls = list(update_ls)
                for i in range(bsz + 1):
                    if i not in elems: # check if there is empty update
                        update_ls.insert(i, torch.tensor([], device="cuda"))
                update_ls = [update_ls[0]] + update_ls[:0:-1]

                if args.sparse_adam:
                    not_touched_ids = update_ls[0]
                    src = torch.zeros((len(not_touched_ids),), dtype=torch.bool, device="cuda")
                    visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_ids, src=src)
                else:
                    visibility_mask = None
                torch.cuda.nvtx.range_pop()

                # HACK: Testing bsz=32/64 for now
                torch.cuda.nvtx.range_push("precompute sums")
                ps_grid_size, ps_blk_size = (64, 256)
                tmp_buffer = torch.empty((bsz-1, ps_grid_size * ps_blk_size), dtype=torch.int, device="cuda") # 31 * #t
                diff_gaussian_rasterization._C.compute_cnt_h(gs_bitmap, tmp_buffer, ps_grid_size, ps_blk_size)
                cnt_d = torch.sum(tmp_buffer, dim=1).flatten()
                filter_len = torch.tensor([len(f) for f in filters], device="cuda")
                cnt_h = filter_len[1:] - cnt_d
                cnt_g = filter_len[:-1] - cnt_d

                torch.cuda.nvtx.range_pop()
                del gs_bitmap, tmp_buffer, filter_len

                torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
                data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
                cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
                cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
                data2cpu_dim = [len(d) for d in data2cpu_ls]
                cat_data2cpu_h.copy_(cat_data2cpu)
                data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
                assert len(data2cpu_ls_h) == bsz + 4
                update_ls_cpu = data2cpu_ls_h[:bsz+1]
                cnt_h = data2cpu_ls_h[-3]
                cnt_d = data2cpu_ls_h[-2]
                cnt_g = data2cpu_ls_h[-1]
                torch.cuda.nvtx.range_pop()

                finish_indices_filters = update_ls_cpu

                assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
                assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians, f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"

                return finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask

            finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask = order_cal_7(filters, batched_cameras)
            
        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
    else:
        visibility_mask = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if args.sparse_adam else None
        
    if args.overlap_cpuadam:
        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz,
                                             sparse_adam=args.sparse_adam,
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        # Create thread for cpuadam
        if overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
    
    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retents = [None for i in range(num_micro_batches)]

    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    with torch.cuda.stream(comm_stream), torch.no_grad():
        this_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        next_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")
        retention_vec = torch.empty((N,), dtype=torch.int32, device="cuda")

        shs_grad = torch.zeros(filters[0].shape[0], 48, device="cuda")
        shs_grad_init_event = torch.cuda.Event()
        shs_grad_init_event.record(comm_stream)

    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream), torch.no_grad():
                # Forward pass
                shs = torch.empty(this_filter_len, 48, device="cuda", requires_grad=True)

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retents[micro_idx] = shs.detach()
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
            
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            shs_retents[micro_idx] = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event


        with torch.cuda.stream(comm_stream), torch.no_grad():
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

                # compute indices on the fly
                if micro_idx == 0:
                    this_bit.scatter_(dim=0, index=filters[micro_idx], src=torch.ones(filters[micro_idx].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                else:
                    this_bit, next_bit = next_bit, this_bit
                    next_bit.scatter_(dim=0, index=filters[micro_idx-1], src=torch.zeros(filters[micro_idx-1].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                
                # NOTE: Here we use `torch.nonzero_static`(torch2.6) instead of `torch.nonzero` to avoid h2d sync.
                # This stems from the need to know #nonzero_elem before cuda kernel launch.
                # When using `torch.nonzero_static`, `size`` need to be a scalar on host, otherwise it falls back to blocking.
                # retention_vec: next index
                retention_vec.scatter_(dim=0, index=filters[micro_idx+1], src=torch.arange(filters[micro_idx+1].shape[0], dtype=torch.int32, device="cuda"))
                # idx_h = torch.nonzero(~this_bit & next_bit).flatten() # torch.nonzero() blocks cpu!!!
                bit_h = ~this_bit & next_bit
                idx_h = torch.empty((cnt_h[micro_idx],), dtype=torch.int64, device="cuda")
                idx_h = torch.nonzero_static(bit_h, size=cnt_h[micro_idx]).flatten()
                host_indices_to_param = idx_h.to(torch.int32)
                param_indices_from_host = torch.gather(retention_vec, dim=0, index=idx_h)
                del idx_h, bit_h
                
                # idx_d = torch.nonzero(this_bit & next_bit).flatten() # overlap # torch.nonzero() blocks cpu!!!
                bit_d = this_bit & next_bit
                idx_d = torch.nonzero_static(bit_d, size=cnt_d[micro_idx]).flatten()
                param_indices_from_rtnt = torch.gather(retention_vec, dim=0, index=idx_d) # reused in gpu2cpu comm
                del bit_d

                # retention_vec: this index
                retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.arange(filters[micro_idx].shape[0], dtype=torch.int32, device="cuda"))
                rtnt_indices_to_param = torch.gather(retention_vec, dim=0, index=idx_d) # reused in gpu2cpu comm
                del idx_d
                
                send_shs2gpu_stream_retention(
                    shs_next, # shs to fill
                    gaussians._parameters, # shs on host
                    shs_retents[micro_idx], # shs from last iter
                    host_indices_to_param,
                    rtnt_indices_to_param,
                    param_indices_from_host,
                    param_indices_from_rtnt,
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                del host_indices_to_param, param_indices_from_host
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")
        filtered_xyz_gpu = torch.gather(gaussians._xyz.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity_gpu = torch.gather(gaussians._opacity.detach(), 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling_gpu = torch.gather(gaussians._scaling.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation_gpu = torch.gather(gaussians._rotation.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)

        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity_gpu)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling_gpu)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation_gpu)
        torch.cuda.nvtx.range_pop()

        # sync event of comm_stream with default_stream to make sure shs has been loaded to gpu
        cpu2gpu_event.wait(default_stream)
        filtered_shs = shs.requires_grad_(False) # this is a view of the original shs.

        # preprocess
        rendered_image, batched_means2D, batched_radiis, batched_colors_detached, dirs  = pipeline_forward_one_step_shs_inplace(filtered_opacity_gpu,
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
        
        shs_grad_init_event.wait(default_stream) # wait for `shs_grad` to finish init`

        v_dirs = spherical_harmonics_bwd_inplace(degrees_to_use=gaussians.active_sh_degree, dirs=dirs, coeffs=filtered_shs.reshape(1, -1, 16, 3), v_coeffs=shs_grad, v_colors=batched_colors_detached.grad)
        dirs.backward(v_dirs)
        torch.cuda.nvtx.range_pop()

        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            gaussians._xyz.grad.scatter_add_(dim=0, src=filtered_xyz_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._opacity.grad.scatter_add_(dim=0, src=_filtered_opacity_gpu.grad, index=this_filter.reshape(-1, 1))
            gaussians._scaling.grad.scatter_add_(dim=0, src=_filtered_scaling_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._rotation.grad.scatter_add_(dim=0, src=_filtered_rotation_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            torch.cuda.nvtx.range_pop()

        del rendered_image, batched_colors_detached, dirs, v_dirs

        # free shs
        shs = None
        del filtered_xyz_gpu, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu, filtered_shs
        del _filtered_opacity_gpu, _filtered_scaling_gpu, _filtered_rotation_gpu

        losses.append(loss.detach())
        del loss

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        if not args.offload_shs_grad_before_every_microbatch:
            if micro_idx < num_micro_batches - 1:
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    # compute indices on the fly

                    rtnt_indices_from_grad = param_indices_from_rtnt
                    grad_indices_to_rtnt = rtnt_indices_to_param

                    # idx_g = torch.nonzero(this_bit & ~next_bit).flatten() # torch.nonzero() blocks cpu!!!
                    bit_g = this_bit & ~next_bit
                    idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()

                    host_indices_from_grad = idx_g.to(torch.int32)
                    grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
                    del idx_g, bit_g

                    # sync event of default_stream with comm_stream
                    gpu2cpu_event.wait(comm_stream)
                    shs_retents[micro_idx] = None
                    shs_grad_next = torch.zeros_like(shs_next, device="cuda")

                    send_shs2cpu_grad_buffer_stream_retention(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        shs_grad_next,
                        host_indices_from_grad,
                        rtnt_indices_from_grad,
                        grad_indices_to_host,
                        grad_indices_to_rtnt,
                        True,
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D
                    )
                    shs_grad = shs_grad_next
                    shs_grad_init_event.record(comm_stream)

                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1
                    
            else:
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    gpu2cpu_event.wait(comm_stream)

                    send_shs2cpu_grad_buffer_stream(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        filters[-1],
                        True,
                        grid_size, block_size
                    )
                    
                    if args.overlap_cpuadam:
                        if args.overlap_cpuadam_version == 3:
                            # set signal to pinned memory to notify gradients have been sent back to cpu
                            diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                            microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

        if args.sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility_mask.scatter_(dim=0, index=filters[micro_idx], src=src)
            torch.cuda.nvtx.range_pop()

    if args.overlap_cpuadam:
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
            cpuadam_worker.start()
        assert args.lr_scale_mode == "sqrt", "Overlap CPUAdam only supports sqrt lr scaling"
        assert args.gpu_cache == "xyzosr", "Overlap CPUAdam only supports xyzosr cache"
        assert not args.stop_update_param, "Overlap CPUAdam does not support stop_update_param"
        # only perform gpu adam
        for param in gaussians.all_parameters()[:4]: # the first 4 parameters are on gpu
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            if args.sparse_adam:
                gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
            else:
                gaussians.optimizer.gpu_adam.step()
        gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)
        cpuadam_worker.join()
        utils.memory_report("after cpuadam_worker joined")
    else:
        torch.cuda.synchronize() # we need to make sure gradients have all been sent back to cpu. 
        gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
            if args.sparse_adam:
                sparse_indices = torch.nonzero(visibility_mask).flatten().to("cpu")
                gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
                gaussians.optimizer.cpu_adam.sparse_adam_inc_step()
                gaussians.optimizer.cpu_adam.sparse_step(sparse_indices=sparse_indices)
            else:
                gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")

    torch.cuda.synchronize()
    return losses, ordered_cams, sparsity



def pipeline_offload_v5_impl(
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
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.offload_shs_grad_before_every_microbatch, "retention 4 currently requires disable offload_shs_grad_before_every_microbatc"

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    with torch.no_grad():
        # prepare all parameters
        xyz_gpu = gaussians.get_xyz
        opacity_gpu_origin = gaussians.get_opacity
        scaling_gpu_origin = gaussians.get_scaling
        rotation_gpu_origin = gaussians.get_rotation

        torch.cuda.nvtx.range_push("calculate_filters")
        # calculate gaussian visible filters for all cameras
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        ) # list of GPU long tensors. len(cameras)
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()

    # Sort cameras using these filters when overlap_cpuadam is enabled.
    overlap_cpuadam_version = args.overlap_cpuadam_version
    order_calculation_version = args.order_calculation_version
    if args.overlap_cpuadam or args.retention != 0:
        torch.cuda.nvtx.range_push("sort cameras")
        
        if order_calculation_version == -1:
        # Computes cpu-adam update list using random order.
            bool_filters = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            bool_filters[camera_ids, gaussian_ids] = 1
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
            last_calc_gaussianids_cpu = last_calc_gaussianids.cpu() # TODO: this is slow
            last_calc_gaussian_ids_per_camera = torch.split(last_calc_gaussianids_cpu, last_calc_percamera_counts_cpu)
            assert sum(last_calc_percamera_counts_cpu) == n_gaussians, "sum(last_calc_percamera_counts_cpu) is supposed to be equal to gaussian_ids.shape[0]"
            
            finish_indices_filters = last_calc_gaussian_ids_per_camera

            if args.sparse_adam:
                src = torch.zeros((len(not_touched_gaussian_ids),), dtype=torch.bool, device="cuda")
                visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_gaussian_ids, src=src)
            else:
                visibility_mask = None
            
        elif order_calculation_version == 4:
        # This version only calculates the order and cpuadam update_ls based on v3
            gs_bitmap = torch.zeros(bsz, n_gaussians, dtype=torch.uint8, device="cuda")
            gs_bitmap[camera_ids, gaussian_ids] = 1
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            
            torch.cuda.nvtx.range_push("init lists")
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            cnt_h = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int64, device="cuda")
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                # col_to_reset = torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                col_to_reset = next_update if i > 1 else torch.nonzero(gs_bitmap[cur_cam] & zero_bitvec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
                col_to_reset = col_to_reset.expand(bsz, -1)
                reset_col_gathered = torch.gather(gs_bitmap, dim=1, index=col_to_reset)
                reset_sum = torch.sum(reset_col_gathered, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam
                next_update = torch.nonzero(gs_bitmap[next_cam] & zero_bitvec).flatten()
                update_ls.append(next_update)
            
            col_to_reset = next_update
            zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
            not_touched_ids = torch.nonzero(zero_bitvec).flatten()

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()
            del not_touched_ids

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]
            sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

            torch.cuda.nvtx.range_push("precompute sums")
            for i in range(bsz-1):
                this_bit = gs_bitmap[ordered_cams[i]]
                next_bit = gs_bitmap[ordered_cams[i+1]]
                cnt_h[i] = torch.sum(~this_bit & next_bit)
                cnt_d[i] = torch.sum(this_bit & next_bit)
                cnt_g[i] = torch.sum(this_bit & ~next_bit)
            torch.cuda.nvtx.range_pop()
            del gs_bitmap

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]

            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
        
        # v5: Optimized the bitmap for memory based on v4.
        elif order_calculation_version == 5:
            match bsz:
                case 4 | 8:
                    dtype = torch.int8
                case 16:
                    dtype = torch.int16
                case 32:
                    dtype = torch.int32
                case 64:
                    dtype = torch.int64
                case _:
                    raise ValueError("Currently supported bsz: (4, 8, 16, 32, 64).")

            torch.cuda.nvtx.range_push("init bitmap and vecs")
            gs_bitmap = torch.zeros((n_gaussians), dtype=dtype, device="cuda")
            for i, f in enumerate(filters):
                gs_bitmap.scatter_add_(dim=0, src=torch.ones((len(f),), dtype=dtype, device="cuda"), index=f)
                if i < bsz - 1:
                    gs_bitmap = gs_bitmap << 1
            
            zero_bitvec = torch.ones((n_gaussians,), dtype=torch.uint8, device="cuda")
            sum_vec = torch.empty((bsz, ), dtype=torch.int32, device="cuda")
            for i in range(bsz):
                sum_vec[i] = len(filters[i])
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("init lists")
            cur_cam = min(enumerate(filters), key=lambda x: len(x[1]))[0] #  make the sparsest sample the last one
            ordered_cams = torch.empty((bsz,), dtype=torch.int32, device="cuda")
            ordered_cams[-1] = cur_cam
            # update_ls = [torch.nonzero(gs_bitmap[cur_cam, :]).flatten()]
            update_ls = [torch.nonzero(gs_bitmap >> (bsz-1-cur_cam) & 1).flatten()]
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("order calculation loop + reverse")
            tmp_vec = torch.empty((n_gaussians,), dtype=torch.uint8, device="cuda")
            for i in range(1, bsz):
                cur_cam = ordered_cams[-i]
                if i > 1:
                    col_to_reset = next_update
                else:
                    diff_gaussian_rasterization._C.map_bit_and_vec(gs_bitmap, zero_bitvec, bsz-1-cur_cam, tmp_vec)
                    col_to_reset = next_update if i > 1 else torch.nonzero(tmp_vec).flatten()
                zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))

                reset_col_gathered = torch.gather(gs_bitmap, dim=0, index=col_to_reset)
                reset_bitmap = torch.empty((bsz, len(col_to_reset)), dtype=torch.uint8, device="cuda")
                diff_gaussian_rasterization._C.extract_reset_bitmap(reset_col_gathered, reset_bitmap)
                reset_sum = torch.sum(reset_bitmap, dim=1)
                sum_vec = sum_vec - reset_sum
                sum_vec[ordered_cams[bsz-i:].squeeze()] = torch.iinfo(torch.int32).max
                next_cam = torch.argmin(sum_vec)
                ordered_cams[-i-1] = next_cam

                diff_gaussian_rasterization._C.map_bit_and_vec(gs_bitmap, zero_bitvec, bsz-1-next_cam, tmp_vec)
                next_update = torch.nonzero(tmp_vec).flatten()
                update_ls.append(next_update)
    
            col_to_reset = next_update
            zero_bitvec.scatter_(dim=0, index=col_to_reset, src=torch.zeros_like(col_to_reset, dtype=torch.uint8))
            not_touched_ids = torch.nonzero(zero_bitvec).flatten()
            if args.sparse_adam:
                src = torch.zeros((len(not_touched_ids),), dtype=torch.bool, device="cuda")
                visibility_mask = torch.ones((n_gaussians,), dtype=torch.bool, device="cuda").scatter_(dim=0, index=not_touched_ids, src=src)
            else:
                visibility_mask = None

            update_ls.append(not_touched_ids)
            update_ls.reverse()
            torch.cuda.nvtx.range_pop()
            del not_touched_ids

            batched_cameras = [batched_cameras[i] for i in ordered_cams]
            filters = [filters[i] for i in ordered_cams]
            sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

            torch.cuda.nvtx.range_push("precompute sums")
            cnt_h = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_d = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            cnt_g = torch.empty((bsz-1,), dtype=torch.int, device="cuda")
            hdg_vec = torch.empty((3, n_gaussians), dtype=torch.uint8, device="cuda")
            for i in range(bsz-1):
                diff_gaussian_rasterization._C.generate_hdg(gs_bitmap, bsz-1-ordered_cams[i], bsz-1-ordered_cams[i+1], hdg_vec)
                cnt_hdg = torch.sum(hdg_vec, dim=1)
                cnt_h[i] = cnt_hdg[0]
                cnt_d[i] = cnt_hdg[1]
                cnt_g[i] = cnt_hdg[2]

            torch.cuda.nvtx.range_pop()
            del gs_bitmap, hdg_vec

            torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
            data2cpu_ls = update_ls + [cnt_h, cnt_d, cnt_g]
            cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
            cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
            data2cpu_dim = [len(d) for d in data2cpu_ls]
            cat_data2cpu_h.copy_(cat_data2cpu)
            data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
            assert len(data2cpu_ls_h) == bsz + 4
            update_ls_cpu = data2cpu_ls_h[:bsz+1]
            cnt_h = data2cpu_ls_h[-3]
            cnt_d = data2cpu_ls_h[-2]
            cnt_g = data2cpu_ls_h[-1]
            torch.cuda.nvtx.range_pop()

            finish_indices_filters = update_ls_cpu

            assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
            assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians, f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"

        else:
            raise ValueError("Invalid order calculation version.")

        torch.cuda.nvtx.range_pop()
    else:
        visibility_mask = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if args.sparse_adam else None
        
    if args.overlap_cpuadam:
        def cpuadam_thread_v3(bsz,
                              n_gaussians,
                              signal_tensor_pinned,
                              finish_indices_filters,
                              cpu_adam,
                              parameters,
                              parameters_grad):
            torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

            version = 3 if args.inplace_zero_grad else 2
            parameters.grad = parameters_grad
            if not args.stop_update_param:
                torch.cuda.nvtx.range_push("cpu_adam.sparse_step()")
                cpu_adam.batched_sparse_step(batch_size=bsz,
                                             batched_sparse_indices=finish_indices_filters,
                                             signal_tensor_pinned=signal_tensor_pinned,
                                             version=version,
                                             scale=1.0/bsz,
                                             sparse_adam=args.sparse_adam,
                                        )
                torch.cuda.nvtx.range_pop()

            if version != 3:
                torch.cuda.nvtx.range_push("cpu_adam:grad.zero_()")
                parameters_grad.zero_() # clear the grad buffer so that it can be reused in the next iteration. 
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        # Create thread for cpuadam
        if overlap_cpuadam_version == 3:
            signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
            microbatch_idx = 0
            cpuadam_worker = threading.Thread(target=cpuadam_thread_v3, args=(bsz,
                                                                              n_gaussians,
                                                                              signal_tensor_pinned,
                                                                              finish_indices_filters,
                                                                              gaussians.optimizer.cpu_adam,
                                                                              gaussians._parameters,
                                                                              parameters_grad_buffer[:n_gaussians, :],
                                                                              ))
            cpuadam_worker.start()
        else:
            raise ValueError("Invalid overlap_cpuadam_version.")            
    
    # accumulate gradients at opacity_gpu, scaling_gpu, rotation_gpu since they are computed afer the activation functions.
    # no need for xyz since it does not have activation function.
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    # declare streams for computationa
    default_stream = torch.cuda.current_stream()

    torch.cuda.synchronize()

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []

    grid_size, block_size = args.grid_size_H, 256
    # grid_size_D, block_size_D = args.grid_size_D, 256
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # load the parameters for the first sample in the batch
        if micro_idx == 0:
            with torch.cuda.stream(comm_stream), torch.no_grad():
                # Forward pass
                shs = torch.empty(this_filter_len, 48, device="cuda")

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs.requires_grad_(True)
                # create an event
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
            
        else:
            shs = shs_next # need to verify that is this the correct way to do this? 
            cpu2gpu_event = next_cpu2gpu_event

        with torch.cuda.stream(comm_stream), torch.no_grad():
            # Forward pass
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

                send_shs2gpu_stream(
                    shs_next,
                    gaussians._parameters,# Why this is a detach? May be this is redundant? 
                    filters[micro_idx+1],
                    grid_size, block_size
                )
                shs_next.requires_grad_(True)
                
                # create an event
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)
        
        with torch.cuda.stream(comm_stream):
            shs_grad = torch.empty_like(shs)

        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")
        filtered_xyz_gpu = torch.gather(gaussians._xyz.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity_gpu = torch.gather(gaussians._opacity.detach(), 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling_gpu = torch.gather(gaussians._scaling.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation_gpu = torch.gather(gaussians._rotation.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)

        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity_gpu)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling_gpu)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation_gpu)
        torch.cuda.nvtx.range_pop()

        # sync event of comm_stream with default_stream to make sure shs has been loaded to gpu
        cpu2gpu_event.wait(default_stream)
        # preprocess
        rendered_image, batched_means2D, batched_radiis = pipeline_forward_one_step(filtered_opacity_gpu,
                                                filtered_scaling_gpu,
                                                filtered_rotation_gpu,
                                                filtered_xyz_gpu,
                                                shs,
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
        shs_grad.copy_(shs.grad)
        shs.grad = None
        del shs

        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            gaussians._xyz.grad.scatter_add_(dim=0, src=filtered_xyz_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._opacity.grad.scatter_add_(dim=0, src=_filtered_opacity_gpu.grad, index=this_filter.reshape(-1, 1))
            gaussians._scaling.grad.scatter_add_(dim=0, src=_filtered_scaling_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._rotation.grad.scatter_add_(dim=0, src=_filtered_rotation_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            torch.cuda.nvtx.range_pop()

        del rendered_image
        del filtered_xyz_gpu, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu
        del _filtered_opacity_gpu, _filtered_scaling_gpu, _filtered_rotation_gpu

        losses.append(loss.detach())
        del loss

        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        with torch.cuda.stream(comm_stream), torch.no_grad():
            gpu2cpu_event.wait(comm_stream)
            send_shs2cpu_grad_buffer_stream(
                shs_grad,
                parameters_grad_buffer[:N, :],
                filters[micro_idx],
                True,
                grid_size, block_size
            )
            
            if args.overlap_cpuadam:
                if args.overlap_cpuadam_version == 3:
                    # set signal to pinned memory to notify gradients have been sent back to cpu
                    diff_gaussian_rasterization._C.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # Update densification state.
        update_densification_stats_pipelineoffload_xyzosr(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

        if args.sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility_mask.scatter_(dim=0, index=filters[micro_idx], src=src)
            torch.cuda.nvtx.range_pop()

    if args.overlap_cpuadam:
        if overlap_cpuadam_version == 3:
            assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
        if overlap_cpuadam_version not in [0, 3]:
            cpuadam_worker.start()
        assert args.lr_scale_mode == "sqrt", "Overlap CPUAdam only supports sqrt lr scaling"
        assert args.gpu_cache == "xyzosr", "Overlap CPUAdam only supports xyzosr cache"
        assert not args.stop_update_param, "Overlap CPUAdam does not support stop_update_param"
        # only perform gpu adam
        for param in gaussians.all_parameters()[:4]: # the first 4 parameters are on gpu
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            if args.sparse_adam:
                gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
            else:
                gaussians.optimizer.gpu_adam.step()
        gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)
        cpuadam_worker.join()
        utils.memory_report("after cpuadam_worker joined")
    else:
        torch.cuda.synchronize() # we need to make sure gradients have all been sent back to cpu. 
        gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]

        timers = utils.get_timers()
        timers.start("grad scale + optimizer step + zero grad")
        for param in gaussians.all_parameters():
            if param.grad is not None:
                param.grad /= args.bsz
        if not args.stop_update_param:
            if args.sparse_adam:
                sparse_indices = torch.nonzero(visibility_mask).flatten().to("cpu")
                gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
                gaussians.optimizer.cpu_adam.sparse_adam_inc_step()
                gaussians.optimizer.cpu_adam.sparse_step(sparse_indices=sparse_indices)
            else:
                gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.parameters_grad_buffer[:N, :].zero_()
        timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()
    return losses

def offload_eval_one_cam(
    camera,
    gaussians,
    background,
    scene
):
    # Prepare parameters.
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation

    filters, _, _ = calculate_filters(
        [camera],
        xyz_gpu,
        opacity_gpu_origin,
        scaling_gpu_origin,
        rotation_gpu_origin
    )

    del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
    this_filter = filters[0]

    filtered_xyz_gpu = torch.gather(xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3))
    filtered_opacity_gpu = torch.gather(gaussians._opacity, 0, this_filter.reshape(-1, 1))
    filtered_scaling_gpu = torch.gather(gaussians._scaling, 0, this_filter.reshape(-1, 1).expand(-1, 3))
    filtered_rotation_gpu = torch.gather(gaussians._rotation, 0, this_filter.reshape(-1, 1).expand(-1, 4))
    
    filtered_opacity_gpu = gaussians.opacity_activation(filtered_opacity_gpu)
    filtered_scaling_gpu = gaussians.scaling_activation(filtered_scaling_gpu)
    filtered_rotation_gpu = gaussians.rotation_activation(filtered_rotation_gpu)

    this_filter_cpu = this_filter.to("cpu")
    filtered_shs_gpu = torch.gather(gaussians._parameters, 0, this_filter_cpu.reshape(-1, 1).expand(-1, 48)).to("cuda")

    # Do rendering.
    rendered_image, _, _ = pipeline_forward_one_step(
        filtered_opacity_gpu=filtered_opacity_gpu,
        filtered_scaling_gpu=filtered_scaling_gpu,
        filtered_rotation_gpu=filtered_rotation_gpu,
        filtered_xyz_gpu=filtered_xyz_gpu,
        filtered_shs=filtered_shs_gpu,
        camera=camera,
        scene=scene,
        gaussians=gaussians,
        background=background,
        pipe_args=None,
        eval=True
    )

    return rendered_image

def braindead_offload_eval_one_cam(
    gaussians,
    scene,
    camera,
    background, 
):
    xyz_gpu = gaussians.get_xyz.to("cuda")
    opacity_gpu = gaussians.get_opacity.to("cuda")
    scaling_gpu = gaussians.get_scaling.to("cuda")
    rotation_gpu = gaussians.get_rotation.to("cuda")
    shs_gpu = gaussians.get_features.to("cuda")

    rendered_image, _, _ = pipeline_forward_one_step(
        filtered_opacity_gpu=opacity_gpu,
        filtered_scaling_gpu=scaling_gpu,
        filtered_rotation_gpu=rotation_gpu,
        filtered_xyz_gpu=xyz_gpu,
        filtered_shs=shs_gpu,
        camera=camera,
        scene=scene,
        gaussians=gaussians,
        background=background,
        pipe_args=None,
        eval=True
    )

    return rendered_image

def sparse_offload_impl(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    sparse_adam=False,
):
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    bsz = len(batched_cameras)
    N = gaussians._xyz.shape[0]

    with torch.no_grad():
        # prepare all parameters
        xyz_gpu = gaussians.get_xyz
        opacity_gpu_origin = gaussians.get_opacity
        scaling_gpu_origin = gaussians.get_scaling
        rotation_gpu_origin = gaussians.get_rotation

        torch.cuda.nvtx.range_push("calculate_filters")
        # calculate gaussian visible filters for all cameras
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        ) # list of GPU long tensors. len(cameras)
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()
    
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    # start the training pipeline
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    visibility = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if sparse_adam else None

    grid_size, block_size = args.grid_size_H, 256
    # grid_size_D, block_size_D = args.grid_size_D, 256

    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        torch.cuda.nvtx.range_push("send shs to gpu")
        filtered_shs = torch.empty(this_filter_len, 48, device="cuda")
        send_shs2gpu_stream(
            filtered_shs,
            gaussians._parameters,
            this_filter,
            grid_size, block_size
        )
        filtered_shs.requires_grad_(True)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")
        filtered_xyz_gpu = torch.gather(gaussians._xyz.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        filtered_opacity_gpu = torch.gather(gaussians._opacity.detach(), 0, this_filter.reshape(-1, 1))
        filtered_scaling_gpu = torch.gather(gaussians._scaling.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3))
        filtered_rotation_gpu = torch.gather(gaussians._rotation.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 4))

        filtered_opacity_gpu = gaussians.opacity_activation(filtered_opacity_gpu).requires_grad_(True)
        filtered_scaling_gpu = gaussians.scaling_activation(filtered_scaling_gpu).requires_grad_(True)
        filtered_rotation_gpu = gaussians.rotation_activation(filtered_rotation_gpu).requires_grad_(True)
        torch.cuda.nvtx.range_pop()

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

        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            gaussians._xyz.grad.scatter_add_(dim=0, src=filtered_xyz_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._opacity.grad.scatter_add_(dim=0, src=filtered_opacity_gpu.grad, index=this_filter.reshape(-1, 1))
            gaussians._scaling.grad.scatter_add_(dim=0, src=filtered_scaling_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._rotation.grad.scatter_add_(dim=0, src=filtered_rotation_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            torch.cuda.nvtx.range_pop()

            del rendered_image
            del filtered_xyz_gpu, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu

            losses.append(loss.detach())
            del loss

            send_shs2cpu_grad_buffer_stream(
                filtered_shs.grad,
                parameters_grad_buffer[:N, :],
                this_filter,
                True,
                grid_size, block_size
            )
            torch.cuda.nvtx.range_pop()

            # Update densification state.
            update_densification_stats_pipelineoffload_xyzosr(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                filters[micro_idx],
                batched_means2D.grad.squeeze(0),
                batched_radiis.squeeze(0),
            )

            if sparse_adam:
                torch.cuda.nvtx.range_push("update visibility")
                src = torch.ones((len(gaussian_ids),), dtype=torch.bool, device="cuda")
                visibility.scatter_(dim=0, index=gaussian_ids, src=src)
                torch.cuda.nvtx.range_pop()

            batched_means2D.grad = None
            del batched_means2D, batched_radiis

        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    gaussians._parameters.grad = gaussians.parameters_grad_buffer[:N, :]

    timers = utils.get_timers()
    timers.start("grad scale + optimizer step + zero grad")
    for param in gaussians.all_parameters():
        if param.grad is not None:
            param.grad /= args.bsz
    if not args.stop_update_param:
        if sparse_adam:
            sparse_indices = torch.nonzero(visibility).flatten().to(torch.int32)
            sparse_indices = sparse_indices.to("cpu")
            gaussians.optimizer.cpu_adam.sparse_adam_inc_step()
            gaussians.optimizer.cpu_adam.sparse_step(sparse_indices=sparse_indices)
            gaussians.optimizer.gpu_adam.step(visibility=visibility)
        else:
            gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.parameters_grad_buffer[:N, :].zero_()
    timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()

    return losses

def fairBraindead_offload_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    sparse_adam=False
):
    args = utils.get_args()
    timers = utils.get_timers()
    losses = []

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    with torch.no_grad():
        torch.cuda.nvtx.range_push("load parameters to gpu")
        xyz_gpu = gaussians._xyz.detach().to("cuda")
        _opacity_gpu = gaussians._opacity.detach().to("cuda")
        _scaling_gpu = gaussians._scaling.detach().to("cuda")
        _rotation_gpu = gaussians._rotation.detach().to("cuda")
        _features_dc_gpu = gaussians._features_dc.detach().to("cuda")
        _features_rest_gpu = gaussians._features_rest.detach().to("cuda")
        sh_degree = gaussians.active_sh_degree
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("activate critical attr")
        opacity_gpu_origin = gaussians.opacity_activation(_opacity_gpu)
        scaling_gpu_origin = gaussians.scaling_activation(_scaling_gpu)
        rotation_gpu_origin = gaussians.rotation_activation(_rotation_gpu)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("calculate_filters")
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        ) # list of GPU long tensors. len(cameras)
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("prealloc pinned memory for grads")
    gaussians._xyz.grad = torch.empty_like(gaussians._xyz, pin_memory=True)
    gaussians._features_dc.grad = torch.empty_like(gaussians._features_dc, pin_memory=True)
    gaussians._features_rest.grad = torch.empty_like(gaussians._features_rest, pin_memory=True)
    gaussians._scaling.grad = torch.empty_like(gaussians._scaling, pin_memory=True)
    gaussians._rotation.grad = torch.empty_like(gaussians._rotation, pin_memory=True)
    gaussians._opacity.grad = torch.empty_like(gaussians._opacity, pin_memory=True)  
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("prealloc space for gpu grads")
    xyz_gpu_grad = torch.zeros_like(xyz_gpu)
    _opacity_gpu_grad = torch.zeros_like(_opacity_gpu)
    _scaling_gpu_grad = torch.zeros_like(_scaling_gpu)
    _rotation_gpu_grad = torch.zeros_like(_rotation_gpu)
    _features_dc_gpu_grad = torch.zeros_like(_features_dc_gpu)
    _features_rest_gpu_grad = torch.zeros_like(_features_rest_gpu)
    torch.cuda.nvtx.range_pop()

    losses = []
    visibility = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if sparse_adam else None

    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro batch {micro_idx}")

        this_filter = filters[micro_idx]

        torch.cuda.nvtx.range_push("prepare filtered parameters")
        filtered_xyz = torch.gather(xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity = torch.gather(_opacity_gpu, 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling = torch.gather(_scaling_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation = torch.gather(_rotation_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)
        
        _filtered_features_dc = torch.gather(_features_dc_gpu.view(-1, 3), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_features_rest = torch.gather(_features_rest_gpu.view(-1, 45), 0, this_filter.reshape(-1, 1).expand(-1, 45)).requires_grad_(True)

        filtered_opacity = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation = gaussians.rotation_activation(_filtered_rotation)
        filtered_shs = torch.cat((_filtered_features_dc, _filtered_features_rest), dim=1)
        torch.cuda.nvtx.range_pop()

        rendered_image, means2D, radiis = pipeline_forward_one_step(
            filtered_opacity,
            filtered_scaling,
            filtered_rotation,
            filtered_xyz,
            filtered_shs,
            camera,
            scene,
            gaussians,
            background,
            None,
            eval=False,
        )
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        loss.backward()
        losses.append(loss.detach())

        # with torch.no_grad():
            # Update densification state.
            #  update_densification_stats_pipelineoffload_xyzosr(
            #     scene,
            #     gaussians,
            #     int(utils.get_img_height()),
            #     int(utils.get_img_width()),
            #     torch.nonzero((radiis > 0)).flatten(),
            #     means2D.grad.squeeze(0),
            #     radiis.squeeze(0),
            # )
        
        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to buffer of origin shape")
            xyz_gpu_grad.scatter_add_(dim=0, src=filtered_xyz.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _opacity_gpu_grad.scatter_add_(dim=0, src=_filtered_opacity.grad, index=this_filter.reshape(-1, 1))
            _scaling_gpu_grad.scatter_add_(dim=0, src=_filtered_scaling.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _rotation_gpu_grad.scatter_add_(dim=0, src=_filtered_rotation.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            _features_dc_gpu_grad.view(-1, 3).scatter_add_(dim=0, src=_filtered_features_dc.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _features_rest_gpu_grad.view(-1, 45).scatter_add_(dim=0, src=_filtered_features_rest.grad, index=this_filter.reshape(-1, 1).expand(-1, 45))
            torch.cuda.nvtx.range_pop()

        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility.scatter_(dim=0, index=filters[micro_idx], src=src)
            del src
            torch.cuda.nvtx.range_pop()
        
        del loss, rendered_image, means2D, radiis
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("send grads back to cpu")
    gaussians._xyz.grad.copy_(xyz_gpu_grad)
    gaussians._opacity.grad.copy_(_opacity_gpu_grad)
    gaussians._scaling.grad.copy_(_scaling_gpu_grad)
    gaussians._rotation.grad.copy_(_rotation_gpu_grad)
    gaussians._features_dc.grad.copy_(_features_dc_gpu_grad)
    gaussians._features_rest.grad.copy_(_features_rest_gpu_grad)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()

    timers.start("grad scale + optimizer step + zero grad")
    for param in gaussians.all_parameters():
        if param.grad is not None:
            param.grad /= args.bsz
    if not args.stop_update_param:
        if sparse_adam:
            sparse_indices = torch.nonzero(visibility).flatten().to(torch.int32)
            sparse_indices = sparse_indices.to("cpu")
            # gaussians.optimizer.sparse_adam_inc_step()
            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
        else:
            gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()  

    return losses, visibility

# braindeatch offload: only load the minimum workload.
def braindeath_offload_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    sparse_adam=False
):
    args = utils.get_args()
    timers = utils.get_timers()
    losses = []

    torch.cuda.nvtx.range_push("alloc pinned memory for grads")
    gaussians._xyz.grad = torch.empty_like(gaussians._xyz, pin_memory=True)
    gaussians._features_dc.grad = torch.empty_like(gaussians._features_dc, pin_memory=True)
    gaussians._features_rest.grad = torch.empty_like(gaussians._features_rest, pin_memory=True)
    gaussians._scaling.grad = torch.empty_like(gaussians._scaling, pin_memory=True)
    gaussians._rotation.grad = torch.empty_like(gaussians._rotation, pin_memory=True)
    gaussians._opacity.grad = torch.empty_like(gaussians._opacity, pin_memory=True)  
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("load parameters from cpu to gpu")
    xyz_gpu = gaussians._xyz.detach().to("cuda").requires_grad_(True)
    _opacity_gpu = gaussians._opacity.detach().to("cuda").requires_grad_(True)
    _scaling_gpu = gaussians._scaling.detach().to("cuda").requires_grad_(True)
    _rotation_gpu = gaussians._rotation.detach().to("cuda").requires_grad_(True)
    _features_dc_gpu = gaussians._features_dc.detach().to("cuda").requires_grad_(True)
    _features_rest_gpu = gaussians._features_rest.detach().to("cuda").requires_grad_(True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("activate parameters")
    opacity_origin = gaussians.opacity_activation(_opacity_gpu)
    scaling_origin = gaussians.scaling_activation(_scaling_gpu)
    rotation_origin = gaussians.rotation_activation(_rotation_gpu)
    shs_origin = torch.cat((_features_dc_gpu, _features_rest_gpu), dim=1)
    torch.cuda.nvtx.range_pop()

    opacity = opacity_origin.detach().requires_grad_(True)
    scaling = scaling_origin.detach().requires_grad_(True)
    rotation = rotation_origin.detach().requires_grad_(True)
    shs = shs_origin.detach().requires_grad_(True)

    visibility = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if sparse_adam else None

    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro batch {micro_idx}")

        rendered_image, means2D, radiis = pipeline_forward_one_step(
            opacity,
            scaling,
            rotation,
            xyz_gpu,
            shs,
            camera,
            scene,
            gaussians,
            background,
            None,
            eval=False,
        )
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        loss.backward()
        losses.append(loss.detach())

        # with torch.no_grad():
            # Update densification state.
            #  update_densification_stats_pipelineoffload_xyzosr(
            #     scene,
            #     gaussians,
            #     int(utils.get_img_height()),
            #     int(utils.get_img_width()),
            #     torch.nonzero((radiis > 0)).flatten(),
            #     means2D.grad.squeeze(0),
            #     radiis.squeeze(0),
            # )

        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            visibility = visibility | (radiis > 0)  
            torch.cuda.nvtx.range_pop()
        
        del loss, rendered_image, means2D, radiis
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward to origin gpu parameters")
    opacity_origin.backward(opacity.grad)
    scaling_origin.backward(scaling.grad)
    rotation_origin.backward(rotation.grad)
    shs_origin.backward(shs.grad)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("send grads back to cpu")
    gaussians._xyz.grad.copy_(xyz_gpu.grad)
    gaussians._opacity.grad.copy_(_opacity_gpu.grad)
    gaussians._scaling.grad.copy_(_scaling_gpu.grad)
    gaussians._rotation.grad.copy_(_rotation_gpu.grad)
    gaussians._features_dc.grad.copy_(_features_dc_gpu.grad)
    gaussians._features_rest.grad.copy_(_features_rest_gpu.grad)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()

    timers.start("grad scale + optimizer step + zero grad")
    for param in gaussians.all_parameters():
        if param.grad is not None:
            param.grad /= args.bsz
    if not args.stop_update_param:
        if sparse_adam:
            sparse_indices = torch.nonzero(visibility).flatten().to(torch.int32)
            sparse_indices = sparse_indices.to("cpu")
            # gaussians.optimizer.sparse_adam_inc_step()
            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
        else:
            gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()  

    return losses, visibility


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
    packed=False,
):
    args = utils.get_args()
    # Prepare camera param.
    # image_width = int(camera.image_width)
    # image_height = int(camera.image_height)
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
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

    if packed:
        camera_ids, gaussian_ids, radiis, means2D, depths, conics, _ = fully_fused_projection(
            means=means3D,  # (N, 3)
            covars=None,
            quats=rotations,
            scales=scales,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            radius_clip=args.radius_clip,
            packed=True,
        )
        if mode == "train":
            means2D.retain_grad()

        camtoworld = torch.inverse(viewmat.unsqueeze(0))
        dirs = means3D[gaussian_ids, :] - camtoworld[0, :3, 3]
        
        colors = spherical_harmonics(
            degrees_to_use=sh_degree,
            dirs=dirs,
            coeffs=shs[gaussian_ids, :, :],
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
            packed=True,
            n_cameras=1,
            camera_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

        rendered_image, _ = rasterize_to_pixels(
            means2d=means2D,
            conics=conics,
            colors=colors,
            opacities=opacities.squeeze(1)[gaussian_ids],
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=background,
            packed=True,
        )
        rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    else:
        radiis, means2D, depths, conics, _ = fully_fused_projection(
            means=means3D,  # (N, 3)
            covars=None,
            quats=rotations,
            scales=scales,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            radius_clip=args.radius_clip,
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
        gaussian_ids = None

    return rendered_image, means2D, radiis, gaussian_ids


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
    pipe_args,
    eval=False,
):
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
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
    assert K.shape == (3, 3)

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    n_selected = filtered_xyz_gpu.shape[0]
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
            width=image_width,
            height=image_height,
            packed=False,
        )
    ) # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    if not eval:
        batched_means2D.retain_grad() # this is only for training. 

    sh_degree = gaussians.active_sh_degree
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    camtoworlds = torch.inverse(viewmat.unsqueeze(0))
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
    


def fairBaseline_accumGrads_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    scaling_modifier=1.0,
    sparse_adam=False,
):
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    assert not args.packed
    assert args.disable_auto_densification

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    with torch.no_grad():
        # prepare all parameters
        xyz_gpu = gaussians.get_xyz
        opacity_gpu_origin = gaussians.get_opacity
        scaling_gpu_origin = gaussians.get_scaling
        rotation_gpu_origin = gaussians.get_rotation
        sh_degree = gaussians.active_sh_degree

        torch.cuda.nvtx.range_push("calculate_filters")
        # calculate gaussian visible filters for all cameras
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        ) # list of GPU long tensors. len(cameras)
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()


    torch.cuda.nvtx.range_push("prealloc space for grads")
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
    gaussians._features_dc.grad = torch.zeros_like(gaussians._features_dc)
    gaussians._features_rest.grad = torch.zeros_like(gaussians._features_rest)
    torch.cuda.nvtx.range_pop()

    losses = []
    visibility = torch.zeros((n_gaussians,), dtype=torch.bool, device="cuda") if sparse_adam else None

    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro idx {micro_idx}")

        this_filter = filters[micro_idx]

        torch.cuda.nvtx.range_push("prepare filtered parameters")
        filtered_xyz = torch.gather(gaussians._xyz.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity = torch.gather(gaussians._opacity.detach(), 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling = torch.gather(gaussians._scaling.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation = torch.gather(gaussians._rotation.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)
        
        _filtered_features_dc = torch.gather(gaussians._features_dc.detach().view(-1, 3), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_features_rest = torch.gather(gaussians._features_rest.detach().view(-1, 45), 0, this_filter.reshape(-1, 1).expand(-1, 45)).requires_grad_(True)

        filtered_opacity = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation = gaussians.rotation_activation(_filtered_rotation)
        filtered_shs = torch.cat((_filtered_features_dc, _filtered_features_rest), dim=1)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("forward")
        rendered_image, means2D, radiis = pipeline_forward_one_step(
            filtered_opacity,
            filtered_scaling,
            filtered_rotation,
            filtered_xyz,
            filtered_shs,
            camera,
            scene,
            gaussians,
            background,
            None
        )
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        losses.append(loss.detach())

        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            gaussians._xyz.grad.scatter_add_(dim=0, src=filtered_xyz.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._opacity.grad.scatter_add_(dim=0, src=_filtered_opacity.grad, index=this_filter.reshape(-1, 1))
            gaussians._scaling.grad.scatter_add_(dim=0, src=_filtered_scaling.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._rotation.grad.scatter_add_(dim=0, src=_filtered_rotation.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            gaussians._features_dc.grad.view(-1, 3).scatter_add_(dim=0, src=_filtered_features_dc.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._features_rest.grad.view(-1, 45).scatter_add_(dim=0, src=_filtered_features_rest.grad, index=this_filter.reshape(-1, 1).expand(-1, 45))
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("update stats")
            # Update densification state.
            update_densification_stats_baseline_accumGrads(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                means2D.grad,
                radiis,
                filters[micro_idx],
            )
            torch.cuda.nvtx.range_pop()

        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility.scatter_(dim=0, index=filters[micro_idx], src=src)
            del src
            torch.cuda.nvtx.range_pop()
        
        del loss, rendered_image, means2D, radiis
        torch.cuda.nvtx.range_pop()

    return losses, visibility


def baseline_accumGrads_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    scaling_modifier=1.0,
    sparse_adam=False,
    packed=False,
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

    visibility = torch.zeros((means3D.shape[0],), dtype=torch.bool, device="cuda") if sparse_adam else None

    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro idx {micro_idx}")
        torch.cuda.nvtx.range_push("forward")
        rendered_image, means2D, radiis, gaussian_ids = baseline_accumGrads_micro_step(
            means3D,
            opacities,
            scales,
            rotations,
            shs,
            sh_degree,
            camera,
            background,
            packed=packed
        )
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        losses.append(loss.detach())

        torch.cuda.nvtx.range_push("update stats")
        with torch.no_grad():
            # Update densification state.
            update_densification_stats_baseline_accumGrads(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                means2D.grad,
                radiis,
                gaussian_ids,
            )
        torch.cuda.nvtx.range_pop()

        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            if packed:
                src = torch.ones((len(gaussian_ids),), dtype=torch.bool, device="cuda")
                visibility.scatter_(dim=0, index=gaussian_ids, src=src)
                del gaussian_ids, src
            else:
                visibility = visibility | (radiis > 0).squeeze()
            
            torch.cuda.nvtx.range_pop()
        
        del loss, rendered_image, means2D, radiis

        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward to origin")
    opacities_origin.backward(opacities.grad)
    scales_origin.backward(scales.grad)
    rotations_origin.backward(rotations.grad)
    shs_origin.backward(shs.grad)
    torch.cuda.nvtx.range_pop()

    return losses, visibility

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    torch.cuda.set_device(args.gpu)
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
            scene = Scene(args, gaussians)
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
    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

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

    if args.log_sampled_filters != 0:
        assert args.retention == 4, "Only implemented for retention v4"
        selected_batches = torch.randperm(min(len(scene.train_cameras_info), args.iterations) // args.bsz, device="cuda", generator=perm_generator)[:args.log_sampled_filter]
        selected_batches = [b * args.bsz + 1 for b in selected_batches]

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
        if args.log_sampled_filters != 0 and iteration in selected_batches:
            log_this_batch_filters = True
        else:
            log_this_batch_filters = False
        
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

            if args.pipelined_offload or args.braindeath_offload:
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

            if args.pipelined_offload or args.braindeath_offload:
                for camera in batched_cameras:
                    camera.world_view_transform = camera.world_view_transform.cuda()
                    # camera.projection_matrix = camera.projection_matrix.cuda()
                    camera.full_proj_transform = camera.full_proj_transform.cuda()

                batched_world_view_transform = []
                for camera in batched_cameras:
                    camera.K = camera.create_k_on_gpu()
                    batched_world_view_transform.append(camera.world_view_transform.transpose(0, 1))
                batched_world_view_transform = torch.stack(batched_world_view_transform)
                batched_world_view_transform_inverse = torch.inverse(batched_world_view_transform)
                batched_world_view_transform_inverse = torch.unbind(batched_world_view_transform_inverse, dim=0)
                for camera, wvt in zip(batched_cameras, batched_world_view_transform_inverse):
                    camera.camtoworlds = wvt.unsqueeze(0)

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
        if args.braindeath_offload:
            N = gaussians._xyz.shape[0]

            if args.fairBraindead:
                losses, visibility = fairBraindead_offload_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    background,
                    sparse_adam=args.sparse_adam,
                )
            else:
                losses, visibility = braindeath_offload_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    background,
                    sparse_adam=args.sparse_adam,
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

        elif args.pipelined_offload:
            assert args.offload, "Pipelined offload requires offloading"
            assert args.bsz > 1, "Pipelined offload requires batch size > 1"
            assert args.gpu_cache == "xyzosr", "Pipelined offload requires xyzosr cache"

            N = gaussians._xyz.shape[0]
            
            if args.pipeline_mode == "no_overlap":
                losses = sparse_offload_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    sparse_adam=args.sparse_adam,
                )
            elif args.retention == 1:
                losses = pipeline_offload_retention_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator
                )
            elif args.retention == 2:
                losses = pipeline_offload_retention_optimized_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator
                )
            elif args.retention == 3:
                losses = pipeline_offload_retention_optimized_v3_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator
                )
            elif args.retention == 4:
                losses, ordered_cams, sparsity = pipeline_offload_retention_optimized_v4_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator,
                    log_this_batch_filters,
                )
            elif args.retention == 5:
                losses, ordered_cams, sparsity = pipeline_offload_retention_optimized_v5_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator,
                    log_this_batch_filters,
                )
            else:
                losses = pipeline_offload_v5_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    gaussians.parameters_grad_buffer,
                    background,
                    pipe_args,
                    comm_stream,
                    perm_generator,
                )
                # losses = pipeline_offload_impl(
                #     gaussians,
                #     scene,
                #     batched_cameras,
                #     gaussians.parameters_grad_buffer,
                #     background,
                #     pipe_args,
                #     comm_stream,
                #     perm_generator
                # )
            batched_screenspace_pkg = {}
            if args.retention == 4 or args.retention == 5:
                batched_cameras = [batched_cameras[i] for i in ordered_cams]
            
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
            if args.retention == 4 or args.retention == 5:
                batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
                log_string = "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
                    iteration,
                    iteration + args.bsz,
                    batched_loss_cpu,
                    sparsity,
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

            if args.fairBaseline:
                losses, visibility = fairBaseline_accumGrads_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    background,
                    sparse_adam=args.sparse_adam,
                )
            else:   
                losses, visibility = baseline_accumGrads_impl(
                    gaussians,
                    scene,
                    batched_cameras,
                    background,
                    sparse_adam=args.sparse_adam,
                    packed=args.packed,
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
                    if args.save_tensors:
                        utils.print_rank_0("NOTE: Saving model as .pt files instead of .ply file.")
                        scene.save_tensors(iteration)
                    else:
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
            if iteration < opt_args.iterations and not args.pipelined_offload and not args.braindeath_offload:
                # utils.memory_report("before optimizer step")
                timers.start("optimizer_step")

                torch.cuda.nvtx.range_push("scale grad")
                if (
                    args.lr_scale_mode != "accumu"
                ):  # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz
                torch.cuda.nvtx.range_pop()

                if not args.stop_update_param:
                    torch.cuda.nvtx.range_push("optimizer step")
                    if args.sparse_adam:
                        if args.braindeath_offload:
                            sparse_indices = torch.nonzero(visibility.squeeze()).flatten().to(torch.int32)
                            sparse_indices = sparse_indices.to("cpu")
                            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
                        else:
                            gaussians.optimizer.step(visibility=visibility)
                        del visibility
                    else:
                        gaussians.optimizer.step()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("zero_grad")
                gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.nvtx.range_pop()
                timers.stop("optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")
                # utils.memory_report("after optimizer step")
                
                if args.offload and not args.braindeath_offload:
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
    del comm_stream
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
                        camera.K = camera.create_k_on_gpu()
                        camera.camtoworlds = torch.inverse(camera.world_view_transform.transpose(0, 1)).unsqueeze(0)

                        if args.braindeath_offload:
                            assert args.offload

                            rendered_image = braindead_offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
                            )
                            batched_image.append(rendered_image)
                        
                        elif args.offload:
                            assert args.gpu_cache == "xyzosr"
                            assert args.backend == "gsplat"
                            rendered_image = offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
                            )
                            batched_image.append(rendered_image)
                            
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
                        camera.K = camera.create_k_on_gpu()
                        camera.camtoworlds = torch.inverse(camera.world_view_transform.transpose(0, 1)).unsqueeze(0)

                        if args.braindeath_offload:
                            assert args.offload

                            rendered_image = braindead_offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
                            )
                            batched_image.append(rendered_image)
                        
                        elif args.offload:
                            assert args.gpu_cache == "xyzosr"
                            assert args.backend == "gsplat"
                            rendered_image = offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
                            )
                            batched_image.append(rendered_image)
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
