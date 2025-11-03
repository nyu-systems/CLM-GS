import os
import torch
import json
from utils.loss_utils import l1_loss
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from scene import Scene, GaussianModel, SceneDataset, TorchSceneDataset, custom_collate_fn
from utils.general_utils import prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
from densification import (
    gsplat_densification, 
    update_densification_stats_pipelineoffload_xyzosr,
    update_densification_stats_baseline_accumGrads,
)
from clm_kernels import (
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    spherical_harmonics_bwd_inplace
)
import clm_kernels
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
from functools import reduce
import fast_tsp
from scene.cameras import get_space_sort_key_dim
import gc
from clm_kernels import fused_ssim

@torch.compile
def loss_combined(image, image_gt, ssim_loss):
    lambda_dssim = 0.2 # TODO: allow this to be set by the user
    Ll1 = l1_loss(image, image_gt)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (
                1.0 - ssim_loss
            )
    return loss

class FusedCompiledLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, image_gt_original):
        image_gt = torch.clamp(image_gt_original / 255.0, 0.0, 1.0)
        ssim_loss = fused_ssim(image.unsqueeze(0), image_gt.unsqueeze(0))
        return loss_combined(image, image_gt, ssim_loss)

FUSED_COMPILED_LOSS_MODULE = FusedCompiledLoss()

def torch_compiled_loss(image, image_gt_original):
    global FUSED_COMPILED_LOSS_MODULE
    loss = FUSED_COMPILED_LOSS_MODULE(image, image_gt_original)
    return loss

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

    # Sort cameras using these filters
    torch.cuda.nvtx.range_push("sort cameras")
    
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
            clm_kernels.scatter_to_bit(gs_bitmap, f, bsz-1-i)

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
            unziped[bsz-1-i] = (sampled_bitmap & 1).to(torch.uint8)
            sampled_bitmap = sampled_bitmap >> 1
        # Compute distance matrix. FIXME: need a better way with less memory
        distance_matrix = (unziped.unsqueeze(1) ^ unziped.unsqueeze(0)).sum(dim=-1).tolist() # intermediate result: (bsz, bsz, n_sampled) = n_gaussians
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("solve order: tsp")
        ordered_cams = fast_tsp.find_tour(distance_matrix, 0.001)
        # find the minimum sparsity camera
        if args.reorder_by_min_sparsity_at_end:
            min_sparsity_i = bsz - 1
            for k in range(0, bsz-1):
                if len(filters[ordered_cams[k]]) < len(filters[ordered_cams[min_sparsity_i]]):
                    min_sparsity_i = k
            ordered_cams = ordered_cams[min_sparsity_i+1:] + ordered_cams[:min_sparsity_i+1]

        torch.cuda.nvtx.range_pop()
        batched_cameras = [batched_cameras[i] for i in ordered_cams]
        filters = [filters[i] for i in ordered_cams]
        sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

        torch.cuda.nvtx.range_push("generate cpuadam update ls")
        # Re-encode the bitmap based on given order
        gs_bitmap.zero_()
        for i, f in enumerate(filters):
            clm_kernels.scatter_to_bit(gs_bitmap, f, bsz-1-i)

        ffs = torch.empty(n_gaussians, dtype=torch.uint8, device="cuda")
        clm_kernels.extract_ffs(gs_bitmap, ffs)
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
        clm_kernels.compute_cnt_h(gs_bitmap, tmp_buffer, ps_grid_size, ps_blk_size)
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
        if args.delay_cpuadam_notaccessed_gs:
            finish_indices_filters = list(finish_indices_filters)

            filter_0 = finish_indices_filters[0] # a tensor
            filter_last = finish_indices_filters[-1] # a tensor
            
            finish_indices_filters[0] = filter_0[:0] # empty tensor
            finish_indices_filters[-1] = torch.cat([filter_last, filter_0], dim=0) # a tensor

            finish_indices_filters = tuple(finish_indices_filters)

        assert len(finish_indices_filters) == bsz + 1, "len(finish_indices_filters) should be equal to bsz + 1"
        assert sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians, f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"

        return finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask

    finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask = order_cal_7(filters, batched_cameras)


    torch.cuda.nvtx.range_pop()

    def cpuadam_thread_v3(bsz,
                            n_gaussians,
                            signal_tensor_pinned,
                            finish_indices_filters,
                            cpu_adam,
                            parameters,
                            parameters_grad):
        torch.cuda.nvtx.range_push(f"cpuadam thread for iter: [{iteration},{iteration+bsz})")

        version = 3 # inplace_zero_grad is true 
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

                    # set signal to pinned memory to notify gradients have been sent back to cpu
                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
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
                    
                    # set signal to pinned memory to notify gradients have been sent back to cpu
                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
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

    assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
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
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trail_default_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_default_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_default_start_event.record(default_stream)

        cpu_adam_trail_comm_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_comm_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_comm_start_event.record(comm_stream)
    cpuadam_worker.join()
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trail_default_end_event.record(default_stream)
        cpu_adam_trail_default_start_event.synchronize()
        cpu_adam_trail_comm_end_event.record(comm_stream)
        cpu_adam_trail_comm_start_event.synchronize()
        args.cpu_adam_trailing_overhead["step"] += 1
        if args.cpu_adam_trailing_overhead["step"] > 3:
            args.cpu_adam_trailing_overhead["from_default_stream"] += cpu_adam_trail_default_start_event.elapsed_time(cpu_adam_trail_default_end_event)
            args.cpu_adam_trailing_overhead["from_comm_stream"] += cpu_adam_trail_comm_start_event.elapsed_time(cpu_adam_trail_comm_end_event)
    utils.memory_report("after cpuadam_worker joined")


    torch.cuda.synchronize()
    return losses, ordered_cams, sparsity


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
    gaussians._xyz.grad = torch.empty_like(gaussians._xyz, pin_memory=args.braindead_pin)
    gaussians._features_dc.grad = torch.empty_like(gaussians._features_dc, pin_memory=args.braindead_pin)
    gaussians._features_rest.grad = torch.empty_like(gaussians._features_rest, pin_memory=args.braindead_pin)
    gaussians._scaling.grad = torch.empty_like(gaussians._scaling, pin_memory=args.braindead_pin)
    gaussians._rotation.grad = torch.empty_like(gaussians._rotation, pin_memory=args.braindead_pin)
    gaussians._opacity.grad = torch.empty_like(gaussians._opacity, pin_memory=args.braindead_pin)  
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
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trailing_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trailing_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trailing_start_event.record()

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
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trailing_end_event.record()
        torch.cuda.synchronize()
        args.cpu_adam_trailing_overhead["step"] += 1
        if args.cpu_adam_trailing_overhead["step"] > 3:
            args.cpu_adam_trailing_overhead["from_default_stream"] += cpu_adam_trailing_start_event.elapsed_time(cpu_adam_trailing_end_event)

    timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()  

    return losses, visibility

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
    

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools
    gc.disable()

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

    # Init background
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
    # declare stream for communication
    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress"
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
            # Load ground-truth images to GPU
            timers.start("load_cameras")
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
            timers.stop("load_cameras")
        if args.braindeath_offload:
            N = gaussians._xyz.shape[0]

            losses, visibility = fairBraindead_offload_impl(
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
            

            losses, ordered_cams, sparsity = pipeline_offload_retention_optimized_v5_impl(
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
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                sparsity,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)

        else:
            raise ValueError("Accumulate grads is not supported")

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
                args.offload,
            )
            end2end_timers.start()

            # Densification
            # utils.memory_report("before densification")
            if args.offload:
                gsplat_densification(
                    iteration, scene, gaussians, batched_screenspace_pkg, offload=args.offload, densify_only=True
                )
            else:  
                raise ValueError("Invalid offload value")
            if not args.disable_auto_densification and iteration <= args.densify_until_iter and iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                iteration, args.bsz, args.densification_interval, 0
            ):
                means3D_all = None
                send2gpu_filter = None
                send2gpu_filter_cpu = None
                    

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
                os.makedirs(save_folder, exist_ok=True)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder
                    + "/chkpnt.pth"
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
    
    if args.log_cpu_adam_trailing_overhead:
        average_cpu_adam_trailing_from_default_stream = (
            args.cpu_adam_trailing_overhead["from_default_stream"] / (args.cpu_adam_trailing_overhead["step"] - 3)
        )
        average_cpu_adam_trailing_from_comm_stream = (
            args.cpu_adam_trailing_overhead["from_comm_stream"] / (args.cpu_adam_trailing_overhead["step"] - 3)
        )

        log_file.write(
            "CPU Adam trailing [from default stream]: {} ms.\n".format(average_cpu_adam_trailing_from_default_stream)
        )
        log_file.write(
            "CPU Adam trailing [from comm stream]: {} ms.\n".format(average_cpu_adam_trailing_from_comm_stream)
        )
        # print("CPU Adam trailing overhead: {} ms.".format(average_cpu_adam_trailing_overhead))



def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, offload=False
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
                for idx in range(1, num_cameras + 1, 1):
                    num_camera_to_load = min(1, num_cameras - idx + 1)
                    batched_cameras = eval_dataset.get_batched_cameras(
                        num_camera_to_load
                    )
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
                            rendered_image = offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
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
                            rendered_image = offload_eval_one_cam(
                                camera=camera,
                                gaussians=scene.gaussians,
                                background=background,
                                scene=scene
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
