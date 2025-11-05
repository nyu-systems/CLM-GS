import os
import torch
import json
from utils.loss_utils import l1_loss
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from scene import Scene, SceneDataset, OffloadSceneDataset
from scene.gaussian_model_fair_naive import GaussianModelBraindeathOffload
from scene.gaussian_model_final import GaussianModelXYZOSROffload
from utils.general_utils import prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
from densification import (
    gsplat_densification, 
    update_densification_stats_pipelineoffload_xyzosr,
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

LAMBDA_DSSIM = 0.2  # Loss weight for SSIM
TILE_SIZE = 16


@torch.compile
def loss_combined(image, image_gt, ssim_loss):
    LAMBDA_DSSIM = 0.2 # TODO: allow this to be set by the user
    Ll1 = l1_loss(image, image_gt)
    loss = (1.0 - LAMBDA_DSSIM) * Ll1 + LAMBDA_DSSIM * (
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
    MICRO_BATCH_SIZE = 1 # NOTE: microbatch here only contains one camera.

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    # K = camera.create_k_on_gpu() # create K now, which may invoke cpu-gpu transfer
    K = camera.K
    n_selected = filtered_xyz_gpu.shape[0]
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

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
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )  # (MICRO_BATCH_SIZE, tile_height, tile_width)

    # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return rendered_image, batched_means2D, batched_radiis, batched_colors_detached, dirs

import threading
import queue
import time

def order_calculation(filters, batched_cameras, n_gaussians, bsz, perm_generator, args):

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

def cpuadam_thread(bsz,
                        n_gaussians,
                        signal_tensor_pinned,
                        finish_indices_filters,
                        cpu_adam,
                        parameters,
                        parameters_grad,
                        iteration,
                        args,
                        ):
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
    """
    Pipeline training with retention-based parameter offloading optimization.
    
    Main idea: Instead of loading all parameters each image render, track which parameters
    are visible across consecutive cameras and retain them on GPU to minimize CPU<->GPU transfers.
    
    The pipeline has 5 stages:
    1. Setup: Calculate visibility filters and sort cameras for optimal caching
    2. Concurrent CPU Adam: Start background thread for CPU-side parameter updates
    3. Initialize training state: Gradients, streams, retention buffers
    4. Main loop: Process each micro-batch with overlapped comm and compute
    5. Finalize: Complete optimizer steps and synchronize
    """
    
    # ============================================================================
    # STAGE 1: SETUP & PREPROCESSING
    # ============================================================================
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    # Calculate which gaussians are visible for each camera
    with torch.no_grad():
        # Get current gaussian parameters
        xyz_gpu = gaussians.get_xyz
        opacity_gpu_origin = gaussians.get_opacity
        scaling_gpu_origin = gaussians.get_scaling
        rotation_gpu_origin = gaussians.get_rotation

        torch.cuda.nvtx.range_push("calculate_filters")
        # Filters: list of indices indicating which gaussians are visible per camera
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        )
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()

    # Sort cameras to maximize gaussian overlap between consecutive frames (better caching)
    torch.cuda.nvtx.range_push("sort cameras")
    finish_indices_filters, batched_cameras, filters, sparsity, ordered_cams, cnt_h, cnt_d, cnt_g, visibility_mask = order_calculation(
        filters, batched_cameras, n_gaussians, bsz, perm_generator, args
    )
    # cnt_h: count of parameters to HOST (load from CPU)
    # cnt_d: count of parameters to DUPLICATE (retain from previous)
    # cnt_g: count of parameters to GARBAGE (offload to CPU)
    torch.cuda.nvtx.range_pop()

    # ============================================================================
    # STAGE 2: CONCURRENT CPU THREAD INITIALIZATION
    # ============================================================================
    # Start a background thread to perform CPU Adam updates asynchronously
    # This overlaps CPU optimization with GPU computation
    signal_tensor_pinned = torch.zeros(bsz, dtype=torch.int32, device="cpu", pin_memory=True)
    microbatch_idx = 0
    cpuadam_worker = threading.Thread(
        target=cpuadam_thread,
        args=(
            bsz,
            n_gaussians,
            signal_tensor_pinned,
            finish_indices_filters,
            gaussians.optimizer.cpu_adam,
            gaussians._parameters,
            parameters_grad_buffer[:n_gaussians, :],
            iteration,
            args,
        )
    )
    cpuadam_worker.start()

    # ============================================================================
    # STAGE 3: TRAINING STATE INITIALIZATION
    # ============================================================================
    # Initialize gradient accumulators for GPU-cached parameters (xyz, opacity, scaling, rotation)
    # These accumulate gradients across all micro-batches before the optimizer step
    gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
    gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
    gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
    gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

    # Stream management: default_stream for compute, comm_stream for CPU<->GPU transfers
    default_stream = torch.cuda.current_stream()

    # Training loop variables
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retents = [None for i in range(num_micro_batches)]  # Retained SH coefficients

    # Kernel launch parameters
    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    # Initialize retention tracking buffers
    # These track which parameters are currently on GPU vs need to be loaded/offloaded
    with torch.cuda.stream(comm_stream), torch.no_grad():
        this_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")  # Current iteration visibility
        next_bit = torch.zeros((N,), dtype=torch.uint8, device="cuda")  # Next iteration visibility
        retention_vec = torch.empty((N,), dtype=torch.int32, device="cuda")  # Index mapping

        shs_grad = torch.zeros(filters[0].shape[0], 48, device="cuda")  # SH gradient buffer
        shs_grad_init_event = torch.cuda.Event()
        shs_grad_init_event.record(comm_stream)

    # ============================================================================
    # STAGE 4: MAIN MICRO-BATCH TRAINING LOOP
    # ============================================================================
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        this_filter = filters[micro_idx]
        this_filter_len = this_filter.shape[0]

        # ------------------------------------------------------------------------
        # 4.1: Load current SH coefficients (CPU → GPU)
        # ------------------------------------------------------------------------
        if micro_idx == 0:
            # First micro-batch: load all visible SH coefficients from CPU
            with torch.cuda.stream(comm_stream), torch.no_grad():
                shs = torch.empty(this_filter_len, 48, device="cuda", requires_grad=True)

                send_shs2gpu_stream(
                    shs,
                    gaussians._parameters,
                    filters[micro_idx],
                    grid_size, block_size
                )
                shs_retents[micro_idx] = shs.detach()
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record(comm_stream)
            
        else:
            # Subsequent micro-batches: reuse SH from previous prefetch
            shs = shs_next
            shs_retents[micro_idx] = shs.detach()
            cpu2gpu_event = next_cpu2gpu_event

        # ------------------------------------------------------------------------
        # 4.2: Prefetch NEXT micro-batch SH coefficients (overlapped with compute)
        # ------------------------------------------------------------------------
        with torch.cuda.stream(comm_stream), torch.no_grad():
            if micro_idx < num_micro_batches - 1:
                shs_next = torch.empty(filters[micro_idx+1].shape[0], 48, device="cuda")

                # Update visibility bitmasks for current and next iterations
                if micro_idx == 0:
                    # Initialize both current and next bitmasks
                    this_bit.scatter_(dim=0, index=filters[micro_idx], src=torch.ones(filters[micro_idx].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                else:
                    # Swap bitmasks and update for next iteration
                    this_bit, next_bit = next_bit, this_bit
                    next_bit.scatter_(dim=0, index=filters[micro_idx-1], src=torch.zeros(filters[micro_idx-1].shape[0], dtype=torch.uint8, device="cuda"))
                    next_bit.scatter_(dim=0, index=filters[micro_idx+1], src=torch.ones(filters[micro_idx+1].shape[0], dtype=torch.uint8, device="cuda"))
                
                # Compute retention indices: classify parameters into 3 categories
                # - Category H (Host): parameters not in current but needed in next (~this_bit & next_bit)
                # - Category D (Duplicate): parameters in both current and next (this_bit & next_bit)
                # - Category G (Garbage): parameters in current but not in next (this_bit & ~next_bit)
                
                # NOTE: Using torch.nonzero_static (torch 2.6+) to avoid device-to-host sync
                # torch.nonzero() would block CPU waiting for GPU, hurting performance
                
                # Setup index mapping for next iteration
                retention_vec.scatter_(dim=0, index=filters[micro_idx+1], src=torch.arange(filters[micro_idx+1].shape[0], dtype=torch.int32, device="cuda"))
                # idx_h = torch.nonzero(~this_bit & next_bit).flatten() # torch.nonzero() blocks cpu!!!
                bit_h = ~this_bit & next_bit
                idx_h = torch.empty((cnt_h[micro_idx],), dtype=torch.int64, device="cuda")
                idx_h = torch.nonzero_static(bit_h, size=cnt_h[micro_idx]).flatten()
                host_indices_to_param = idx_h.to(torch.int32)  # Parameter indices to load from host
                param_indices_from_host = torch.gather(retention_vec, dim=0, index=idx_h)  # Where to put them
                del idx_h, bit_h
                
                # idx_d = torch.nonzero(this_bit & next_bit).flatten() # overlap # torch.nonzero() blocks cpu!!!
                bit_d = this_bit & next_bit
                idx_d = torch.nonzero_static(bit_d, size=cnt_d[micro_idx]).flatten()
                param_indices_from_rtnt = torch.gather(retention_vec, dim=0, index=idx_d) # reused in gpu2cpu comm
                del bit_d

                # Update retention_vec for current iteration mapping
                retention_vec.scatter_(dim=0, index=filters[micro_idx], src=torch.arange(filters[micro_idx].shape[0], dtype=torch.int32, device="cuda"))
                rtnt_indices_to_param = torch.gather(retention_vec, dim=0, index=idx_d)  # Current iteration indices
                del idx_d
                
                # Transfer SH coefficients: mix of retained (from GPU) and loaded (from CPU)
                send_shs2gpu_stream_retention(
                    shs_next,                      # Output: SH for next iteration
                    gaussians._parameters,         # Input: SH on host (CPU)
                    shs_retents[micro_idx],       # Input: SH retained from current iteration (GPU)
                    host_indices_to_param,         # Category H: param → next mapping
                    rtnt_indices_to_param,         # Category D: current → next mapping
                    param_indices_from_host,       # Category H: indices for CPU load
                    param_indices_from_rtnt,       # Category D: indices for GPU retention
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_next.requires_grad_(True)
                del host_indices_to_param, param_indices_from_host
                
                next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                next_cpu2gpu_event.record(comm_stream)

        # ------------------------------------------------------------------------
        # 4.3: Forward pass - Render image with filtered gaussian parameters
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")
        
        # Gather only the visible gaussians for this camera (reduces computation)
        filtered_xyz_gpu = torch.gather(gaussians._xyz.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity_gpu = torch.gather(gaussians._opacity.detach(), 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling_gpu = torch.gather(gaussians._scaling.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation_gpu = torch.gather(gaussians._rotation.detach(), 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)

        # Apply activation functions to constrain parameter ranges
        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity_gpu)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling_gpu)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation_gpu)
        torch.cuda.nvtx.range_pop()

        # Wait for SH coefficients to finish loading from CPU
        cpu2gpu_event.wait(default_stream)
        filtered_shs = shs.requires_grad_(False)

        # Render image using filtered gaussian splatting
        rendered_image, batched_means2D, batched_radiis, batched_colors_detached, dirs = pipeline_forward_one_step_shs_inplace(
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_xyz_gpu,
            filtered_shs,
            batched_cameras[micro_idx],
            scene,
            gaussians,
            background,
            pipe_args
        )

        # Compute loss
        loss = torch_compiled_loss(rendered_image, batched_cameras[micro_idx].original_image)
        torch.cuda.nvtx.range_pop()
        
        # ------------------------------------------------------------------------
        # 4.4: Backward pass - Compute gradients
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("backward_pass")
        loss.backward()
        
        # Wait for shs_grad buffer to be ready
        shs_grad_init_event.wait(default_stream)

        # Manual backward for spherical harmonics (custom gradient computation)
        v_dirs = spherical_harmonics_bwd_inplace(
            degrees_to_use=gaussians.active_sh_degree,
            dirs=dirs,
            coeffs=filtered_shs.reshape(1, -1, 16, 3),
            v_coeffs=shs_grad,
            v_colors=batched_colors_detached.grad
        )
        dirs.backward(v_dirs)
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.5: Accumulate gradients back to full parameter tensors
        # ------------------------------------------------------------------------
        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            # Scatter filtered gradients back to full parameter gradient buffers
            gaussians._xyz.grad.scatter_add_(dim=0, src=filtered_xyz_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._opacity.grad.scatter_add_(dim=0, src=_filtered_opacity_gpu.grad, index=this_filter.reshape(-1, 1))
            gaussians._scaling.grad.scatter_add_(dim=0, src=_filtered_scaling_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            gaussians._rotation.grad.scatter_add_(dim=0, src=_filtered_rotation_gpu.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            torch.cuda.nvtx.range_pop()

        # Cleanup temporary tensors
        del rendered_image, batched_colors_detached, dirs, v_dirs
        shs = None
        del filtered_xyz_gpu, filtered_opacity_gpu, filtered_scaling_gpu, filtered_rotation_gpu, filtered_shs
        del _filtered_opacity_gpu, _filtered_scaling_gpu, _filtered_rotation_gpu

        losses.append(loss.detach())
        del loss

        # Mark completion of GPU computation for this micro-batch
        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)


        # ------------------------------------------------------------------------
        # 4.6: Offload SH gradients back to CPU (GPU → CPU)
        # ------------------------------------------------------------------------
        if micro_idx < num_micro_batches - 1:
            # Non-final micro-batch: use retention-based selective gradient offloading
            with torch.cuda.stream(comm_stream), torch.no_grad():
                # Reuse indices from prefetch step (Category D and G)
                rtnt_indices_from_grad = param_indices_from_rtnt  # Category D: retained indices
                grad_indices_to_rtnt = rtnt_indices_to_param

                # idx_g = torch.nonzero(this_bit & ~next_bit).flatten() # torch.nonzero() blocks cpu!!!
                bit_g = this_bit & ~next_bit
                idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()
                host_indices_from_grad = idx_g.to(torch.int32)
                grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
                del idx_g, bit_g

                # Wait for backward pass to complete
                gpu2cpu_event.wait(comm_stream)
                shs_retents[micro_idx] = None
                shs_grad_next = torch.zeros_like(shs_next, device="cuda")

                # Offload gradients: mix of retained (keep on GPU) and offloaded (send to CPU)
                send_shs2cpu_grad_buffer_stream_retention(
                    shs_grad,                      # Input: current SH gradients
                    parameters_grad_buffer[:N, :], # Output: CPU gradient buffer
                    shs_grad_next,                 # Output: next iteration SH gradients (retained on GPU)
                    host_indices_from_grad,        # Category G: indices to offload to CPU
                    rtnt_indices_from_grad,        # Category D: indices to retain on GPU
                    grad_indices_to_host,          # Category G: mapping to CPU buffer
                    grad_indices_to_rtnt,          # Category D: mapping to next grad buffer
                    True,
                    grid_size,
                    block_size,
                    grid_size_D,
                    block_size_D
                )
                shs_grad = shs_grad_next
                shs_grad_init_event.record(comm_stream)

                # Signal CPU Adam thread that gradients are ready for this micro-batch
                clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                microbatch_idx += 1
                
        else:
            # Final micro-batch: offload all gradients to CPU
            with torch.cuda.stream(comm_stream), torch.no_grad():
                gpu2cpu_event.wait(comm_stream)

                send_shs2cpu_grad_buffer_stream(
                    shs_grad,
                    parameters_grad_buffer[:N, :],
                    filters[-1],
                    True,
                    grid_size, block_size
                )
                
                # Signal CPU Adam thread that final gradients are ready
                clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                microbatch_idx += 1

        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.7: Update densification statistics (for adaptive gaussian control)
        # ------------------------------------------------------------------------
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

        # Update visibility mask for sparse Adam optimizer (tracks which parameters received gradients)
        if args.sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility_mask.scatter_(dim=0, index=filters[micro_idx], src=src)
            torch.cuda.nvtx.range_pop()

    # ============================================================================
    # STAGE 5: POST-TRAINING OPTIMIZATION & CLEANUP
    # ============================================================================
    
    # Sanity checks for overlapped CPU Adam configuration
    assert microbatch_idx == bsz, "microbatch_idx should be equal to bsz."
    assert args.lr_scale_mode == "sqrt", "Overlap CPUAdam only supports sqrt lr scaling"
    assert not args.stop_update_param, "Overlap CPUAdam does not support stop_update_param"
    
    # ------------------------------------------------------------------------
    # 5.1: GPU Adam optimizer step (for xyz, opacity, scaling, rotation)
    # ------------------------------------------------------------------------
    # Average gradients across batch size
    for param in gaussians.all_parameters()[:4]:  # First 4 parameters are cached on GPU
        if param.grad is not None:
            param.grad /= args.bsz
    
    # Apply optimizer step
    if not args.stop_update_param:
        if args.sparse_adam:
            # Sparse Adam: only update parameters that received gradients
            gaussians.optimizer.gpu_adam.step(visibility=visibility_mask)
        else:
            # Dense Adam: update all parameters
            gaussians.optimizer.gpu_adam.step()
    gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)
    
    # ------------------------------------------------------------------------
    # 5.2: Wait for concurrent CPU Adam thread to complete
    # ------------------------------------------------------------------------
    # Optional: measure trailing overhead (time spent waiting for CPU Adam)
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trail_default_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_default_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_default_start_event.record(default_stream)

        cpu_adam_trail_comm_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_comm_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trail_comm_start_event.record(comm_stream)
    
    # Block until CPU Adam thread finishes processing all micro-batches
    cpuadam_worker.join()
    
    # Log CPU Adam trailing overhead if enabled
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trail_default_end_event.record(default_stream)
        cpu_adam_trail_default_start_event.synchronize()
        cpu_adam_trail_comm_end_event.record(comm_stream)
        cpu_adam_trail_comm_start_event.synchronize()
        args.cpu_adam_trailing_overhead["step"] += 1
        if args.cpu_adam_trailing_overhead["step"] > 3:  # Skip first 3 warmup steps
            args.cpu_adam_trailing_overhead["from_default_stream"] += cpu_adam_trail_default_start_event.elapsed_time(cpu_adam_trail_default_end_event)
            args.cpu_adam_trailing_overhead["from_comm_stream"] += cpu_adam_trail_comm_start_event.elapsed_time(cpu_adam_trail_comm_end_event)
    
    utils.memory_report("after cpuadam_worker joined")

    # ------------------------------------------------------------------------
    # 5.3: Final synchronization and return
    # ------------------------------------------------------------------------
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
    """
    Simple "braindead" offload implementation for baseline comparison.
    
    Strategy: Load ALL parameters to GPU once, process all cameras sequentially,
    then offload ALL gradients back to CPU. This is simpler but less memory-efficient
    than retention-based approaches like pipeline_offload_retention_optimized_v5.
    
    The pipeline has 4 stages:
    1. Setup: Load all parameters to GPU and calculate visibility filters
    2. Initialize: Allocate gradient buffers on both CPU (pinned) and GPU
    3. Main loop: Process each camera sequentially, accumulating gradients on GPU
    4. Finalize: Offload all gradients to CPU and perform optimizer step
    
    This serves as a baseline to measure the effectiveness of more sophisticated
    retention-based parameter offloading strategies.
    """
    
    # ============================================================================
    # STAGE 1: SETUP - LOAD PARAMETERS & CALCULATE VISIBILITY
    # ============================================================================
    args = utils.get_args()
    timers = utils.get_timers()
    losses = []

    bsz = len(batched_cameras)
    n_gaussians = gaussians._xyz.shape[0]

    with torch.no_grad():
        # Load ALL gaussian parameters from CPU to GPU at once
        # Unlike retention-based methods, we don't selectively load parameters per camera
        torch.cuda.nvtx.range_push("load parameters to gpu")
        xyz_gpu = gaussians._xyz.detach().to("cuda")
        _opacity_gpu = gaussians._opacity.detach().to("cuda")
        _scaling_gpu = gaussians._scaling.detach().to("cuda")
        _rotation_gpu = gaussians._rotation.detach().to("cuda")
        _features_dc_gpu = gaussians._features_dc.detach().to("cuda")      # SH DC coefficients
        _features_rest_gpu = gaussians._features_rest.detach().to("cuda")  # SH higher-order coefficients
        sh_degree = gaussians.active_sh_degree
        torch.cuda.nvtx.range_pop()

        # Apply activation functions to transform parameters to valid ranges
        torch.cuda.nvtx.range_push("activate critical attr")
        opacity_gpu_origin = gaussians.opacity_activation(_opacity_gpu)      # → [0, 1]
        scaling_gpu_origin = gaussians.scaling_activation(_scaling_gpu)      # → positive values
        rotation_gpu_origin = gaussians.rotation_activation(_rotation_gpu)   # → normalized quaternions
        torch.cuda.nvtx.range_pop()

        # Calculate which gaussians are visible for each camera (frustum culling)
        # This determines which subset of gaussians to render for each camera
        torch.cuda.nvtx.range_push("calculate_filters")
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin
        )  # Returns: list of index tensors, one per camera
        del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
        torch.cuda.nvtx.range_pop()

    # ============================================================================
    # STAGE 2: INITIALIZE GRADIENT BUFFERS
    # ============================================================================
    # Allocate pinned memory on CPU for fast GPU→CPU transfer at the end
    # Pinned memory enables asynchronous transfers and better bandwidth
    torch.cuda.nvtx.range_push("prealloc pinned memory for grads")
    gaussians._xyz.grad = torch.empty_like(gaussians._xyz, pin_memory=True)
    gaussians._features_dc.grad = torch.empty_like(gaussians._features_dc, pin_memory=True)
    gaussians._features_rest.grad = torch.empty_like(gaussians._features_rest, pin_memory=True)
    gaussians._scaling.grad = torch.empty_like(gaussians._scaling, pin_memory=True)
    gaussians._rotation.grad = torch.empty_like(gaussians._rotation, pin_memory=True)
    gaussians._opacity.grad = torch.empty_like(gaussians._opacity, pin_memory=True)  
    torch.cuda.nvtx.range_pop()

    # Allocate gradient accumulation buffers on GPU
    # These accumulate gradients from all cameras before bulk transfer to CPU
    torch.cuda.nvtx.range_push("prealloc space for gpu grads")
    xyz_gpu_grad = torch.zeros_like(xyz_gpu)
    _opacity_gpu_grad = torch.zeros_like(_opacity_gpu)
    _scaling_gpu_grad = torch.zeros_like(_scaling_gpu)
    _rotation_gpu_grad = torch.zeros_like(_rotation_gpu)
    _features_dc_gpu_grad = torch.zeros_like(_features_dc_gpu)
    _features_rest_gpu_grad = torch.zeros_like(_features_rest_gpu)
    torch.cuda.nvtx.range_pop()

    losses = []
    # Visibility mask tracks which gaussians received gradients (for sparse Adam)
    visibility = torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda") if sparse_adam else None

    # ============================================================================
    # STAGE 3: MAIN CAMERA PROCESSING LOOP
    # ============================================================================
    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro batch {micro_idx}")

        this_filter = filters[micro_idx]  # Indices of visible gaussians for this camera

        # ------------------------------------------------------------------------
        # 3.1: Gather visible parameters for this camera
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("prepare filtered parameters")
        
        # Gather only the visible gaussians (reduces computation and memory)
        filtered_xyz = torch.gather(xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_opacity = torch.gather(_opacity_gpu, 0, this_filter.reshape(-1, 1)).requires_grad_(True)
        _filtered_scaling = torch.gather(_scaling_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_rotation = torch.gather(_rotation_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 4)).requires_grad_(True)
        
        # Gather spherical harmonics features (3 channels for DC, 45 for higher-order terms)
        _filtered_features_dc = torch.gather(_features_dc_gpu.view(-1, 3), 0, this_filter.reshape(-1, 1).expand(-1, 3)).requires_grad_(True)
        _filtered_features_rest = torch.gather(_features_rest_gpu.view(-1, 45), 0, this_filter.reshape(-1, 1).expand(-1, 45)).requires_grad_(True)

        # Apply activation functions and prepare SH features
        filtered_opacity = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation = gaussians.rotation_activation(_filtered_rotation)
        filtered_shs = torch.cat((_filtered_features_dc, _filtered_features_rest), dim=1)  # Combine DC + rest
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 3.2: Forward pass - Render image
        # ------------------------------------------------------------------------
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
        
        # ------------------------------------------------------------------------
        # 3.3: Backward pass - Compute loss and gradients
        # ------------------------------------------------------------------------
        loss = torch_compiled_loss(rendered_image, camera.original_image)
        loss.backward()
        losses.append(loss.detach())

        with torch.no_grad():
            # Update densification state.
            # import pdb; pdb.set_trace()
            update_densification_stats_pipelineoffload_xyzosr(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                this_filter,# on gpu
                means2D.grad.squeeze(0),# on gpu
                radiis.squeeze(0), # on gpu
            )
        
        # ------------------------------------------------------------------------
        # 3.4: Accumulate gradients back to full-size buffers
        # ------------------------------------------------------------------------
        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to buffer of origin shape")
            # Scatter filtered gradients back to full gradient buffers (accumulation)
            xyz_gpu_grad.scatter_add_(dim=0, src=filtered_xyz.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _opacity_gpu_grad.scatter_add_(dim=0, src=_filtered_opacity.grad, index=this_filter.reshape(-1, 1))
            _scaling_gpu_grad.scatter_add_(dim=0, src=_filtered_scaling.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _rotation_gpu_grad.scatter_add_(dim=0, src=_filtered_rotation.grad, index=this_filter.reshape(-1, 1).expand(-1, 4))
            _features_dc_gpu_grad.view(-1, 3).scatter_add_(dim=0, src=_filtered_features_dc.grad, index=this_filter.reshape(-1, 1).expand(-1, 3))
            _features_rest_gpu_grad.view(-1, 45).scatter_add_(dim=0, src=_filtered_features_rest.grad, index=this_filter.reshape(-1, 1).expand(-1, 45))
            torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 3.5: Update visibility mask (for sparse Adam)
        # ------------------------------------------------------------------------
        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            # Mark which gaussians received gradients in this iteration
            src = torch.ones((len(filters[micro_idx]),), dtype=torch.bool, device="cuda")
            visibility.scatter_(dim=0, index=filters[micro_idx], src=src)
            del src
            torch.cuda.nvtx.range_pop()
        
        # Cleanup temporary tensors
        del loss, rendered_image, means2D, radiis
        torch.cuda.nvtx.range_pop()

    # ============================================================================
    # STAGE 4: FINALIZE - GRADIENT OFFLOAD & OPTIMIZER STEP
    # ============================================================================
    
    # ------------------------------------------------------------------------
    # 4.1: Bulk transfer all gradients from GPU back to CPU
    # ------------------------------------------------------------------------
    torch.cuda.nvtx.range_push("send grads back to cpu")
    # Copy accumulated gradients from GPU to pinned CPU memory
    # This is a single bulk transfer rather than per-camera transfers
    gaussians._xyz.grad.copy_(xyz_gpu_grad)
    gaussians._opacity.grad.copy_(_opacity_gpu_grad)
    gaussians._scaling.grad.copy_(_scaling_gpu_grad)
    gaussians._rotation.grad.copy_(_rotation_gpu_grad)
    gaussians._features_dc.grad.copy_(_features_dc_gpu_grad)
    gaussians._features_rest.grad.copy_(_features_rest_gpu_grad)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()  # Wait for all GPU operations to complete

    # ------------------------------------------------------------------------
    # 4.2: Perform optimizer step on CPU
    # ------------------------------------------------------------------------
    timers.start("grad scale + optimizer step + zero grad")
    
    # Optional: measure time spent on CPU Adam optimization
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trailing_start_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trailing_end_event = torch.cuda.Event(enable_timing=True)
        cpu_adam_trailing_start_event.record()

    # Average gradients across batch size
    for param in gaussians.all_parameters():
        if param.grad is not None:
            param.grad /= args.bsz
    
    # Apply optimizer update
    if not args.stop_update_param:
        if sparse_adam:
            # Sparse Adam: only update parameters that received gradients
            sparse_indices = torch.nonzero(visibility).flatten().to(torch.int32)
            sparse_indices = sparse_indices.to("cpu")
            # gaussians.optimizer.sparse_adam_inc_step()
            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
        else:
            # Dense Adam: update all parameters
            gaussians.optimizer.step()
    
    gaussians.optimizer.zero_grad(set_to_none=True)
    
    # Log CPU Adam overhead if enabled
    if args.log_cpu_adam_trailing_overhead:
        cpu_adam_trailing_end_event.record()
        torch.cuda.synchronize()
        args.cpu_adam_trailing_overhead["step"] += 1
        if args.cpu_adam_trailing_overhead["step"] > 3:  # Skip first 3 warmup steps
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
    MICRO_BATCH_SIZE = 1 # NOTE: microbatch here only contains one camera.
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
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )  # (MICRO_BATCH_SIZE, tile_height, tile_width)


    # Rasterize to pixels. batched_rendered_image: (MICRO_BATCH_SIZE, image_height, image_width, 3)
    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return rendered_image, batched_means2D, batched_radiis
    

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
    if args.braindeath_offload:
        gaussians = GaussianModelBraindeathOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelBraindeathOffload")
        log_file.write("Using GaussianModelBraindeathOffload\n")
    elif args.final_offload:
        gaussians = GaussianModelXYZOSROffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelXYZOSROffload")
        log_file.write("Using GaussianModelXYZOSROffload\n")
    else:
        raise ValueError(f"Invalid offload configuration: braindeath_offload={args.braindeath_offload}, final_offload={args.final_offload}")

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
        if args.braindeath_offload:
            # BASELINE: Simple bulk parameter transfer strategy
            # Load all params → process all cameras → offload all gradients
            N = gaussians._xyz.shape[0]

            losses, visibility = fairBraindead_offload_impl(
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

        elif args.final_offload:
            # OPTIMIZED: Retention-based parameter offloading with overlapped comm/compute
            # Selective loading → retention across cameras → concurrent CPU Adam
            assert args.bsz > 1, "Pipelined offload requires batch size > 1"

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

        else:
            raise ValueError("Accumulate grads is not supported")

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
            # if iteration in [8797,8798,8799,8800]:
            #     import pdb; pdb.set_trace()
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
            if iteration < opt_args.iterations and not args.final_offload and not args.braindeath_offload:
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
                        if args.braindeath_offload:
                            sparse_indices = torch.nonzero(visibility.squeeze()).flatten().to(torch.int32)
                            sparse_indices = sparse_indices.to("cpu")
                            gaussians.optimizer.sparse_step(sparse_indices=sparse_indices)
                        elif args.final_offload:
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
                if args.final_offload:
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
            num_cameras = config["num_cameras"]
            eval_dataset = OffloadSceneDataset(config["cameras_info"])
            # Init dataloader: num_workers = 0
            dataloader = DataLoader(
                eval_dataset,
                batch_size=1,
                # shuffle=True,
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

                    if args.braindeath_offload:

                        rendered_image = braindead_offload_eval_one_cam(
                            camera=camera,
                            gaussians=scene.gaussians,
                            background=background,
                            scene=scene
                        )
                        batched_image.append(rendered_image)
                    
                    elif args.final_offload:
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
