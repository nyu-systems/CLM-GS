
import torch
import math
import threading
import utils.general_utils as utils
import clm_kernels
import fast_tsp
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from clm_kernels import (
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    spherical_harmonics_bwd_inplace
)
from densification import update_densification_stats_pipelineoffload_xyzosr
from strategies.base_engine import torch_compiled_loss, TILE_SIZE, calculate_filters, pipeline_forward_one_step


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

def clm_offload_train_one_batch(
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


def clm_offload_eval_one_cam(
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


    
