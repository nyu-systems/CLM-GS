import torch
import math
import utils.general_utils as utils
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from densification import update_densification_stats_offload_accum_grads
from strategies.base_engine import (
    torch_compiled_loss,
    TILE_SIZE,
    calculate_filters,
    pipeline_forward_one_step,
)


def naive_offload_eval_one_cam(
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
        eval=True,
    )

    return rendered_image


def naive_offload_train_one_batch(
    gaussians, scene, batched_cameras, background, sparse_adam=False
):
    """
    A naive offload implementation for 3DGS.

    Strategy: Load ALL parameters to GPU once, process all cameras sequentially,
    then offload ALL gradients back to CPU. 

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
        # Unlike CLM_offload, we don't selectively load parameters per camera
        torch.cuda.nvtx.range_push("load parameters to gpu")
        xyz_gpu = gaussians._xyz.detach().to("cuda")
        _opacity_gpu = gaussians._opacity.detach().to("cuda")
        _scaling_gpu = gaussians._scaling.detach().to("cuda")
        _rotation_gpu = gaussians._rotation.detach().to("cuda")
        _features_dc_gpu = gaussians._features_dc.detach().to(
            "cuda"
        )  # SH DC coefficients
        _features_rest_gpu = gaussians._features_rest.detach().to(
            "cuda"
        )  # SH higher-order coefficients
        sh_degree = gaussians.active_sh_degree
        torch.cuda.nvtx.range_pop()

        # Apply activation functions to transform parameters to valid ranges
        torch.cuda.nvtx.range_push("activate critical attr")
        opacity_gpu_origin = gaussians.opacity_activation(_opacity_gpu)  # → [0, 1]
        scaling_gpu_origin = gaussians.scaling_activation(
            _scaling_gpu
        )  # → positive values
        rotation_gpu_origin = gaussians.rotation_activation(
            _rotation_gpu
        )  # → normalized quaternions
        torch.cuda.nvtx.range_pop()

        # Calculate which gaussians are visible for each camera (frustum culling)
        # This determines which subset of gaussians to render for each camera
        torch.cuda.nvtx.range_push("calculate_filters")
        filters, camera_ids, gaussian_ids = calculate_filters(
            batched_cameras,
            xyz_gpu,
            opacity_gpu_origin,
            scaling_gpu_origin,
            rotation_gpu_origin,
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
    gaussians._features_dc.grad = torch.empty_like(
        gaussians._features_dc, pin_memory=True
    )
    gaussians._features_rest.grad = torch.empty_like(
        gaussians._features_rest, pin_memory=True
    )
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
    visibility = (
        torch.zeros((xyz_gpu.shape[0],), dtype=torch.bool, device="cuda")
        if sparse_adam
        else None
    )

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
        filtered_xyz = torch.gather(
            xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)
        ).requires_grad_(True)
        _filtered_opacity = torch.gather(
            _opacity_gpu, 0, this_filter.reshape(-1, 1)
        ).requires_grad_(True)
        _filtered_scaling = torch.gather(
            _scaling_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)
        ).requires_grad_(True)
        _filtered_rotation = torch.gather(
            _rotation_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 4)
        ).requires_grad_(True)

        # Gather spherical harmonics features (3 channels for DC, 45 for higher-order terms)
        _filtered_features_dc = torch.gather(
            _features_dc_gpu.view(-1, 3), 0, this_filter.reshape(-1, 1).expand(-1, 3)
        ).requires_grad_(True)
        _filtered_features_rest = torch.gather(
            _features_rest_gpu.view(-1, 45),
            0,
            this_filter.reshape(-1, 1).expand(-1, 45),
        ).requires_grad_(True)

        # Apply activation functions and prepare SH features
        filtered_opacity = gaussians.opacity_activation(_filtered_opacity)
        filtered_scaling = gaussians.scaling_activation(_filtered_scaling)
        filtered_rotation = gaussians.rotation_activation(_filtered_rotation)
        filtered_shs = torch.cat(
            (_filtered_features_dc, _filtered_features_rest), dim=1
        )  # Combine DC + rest
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
            update_densification_stats_offload_accum_grads(
                scene,
                gaussians,
                int(utils.get_img_height()),
                int(utils.get_img_width()),
                this_filter,  # on gpu
                means2D.grad.squeeze(0),  # on gpu
                radiis.squeeze(0),  # on gpu
            )

        # ------------------------------------------------------------------------
        # 3.4: Accumulate gradients back to full-size buffers
        # ------------------------------------------------------------------------
        with torch.no_grad():
            torch.cuda.nvtx.range_push(
                "scatter gpu grads back to buffer of origin shape"
            )
            # Scatter filtered gradients back to full gradient buffers (accumulation)
            xyz_gpu_grad.scatter_add_(
                dim=0,
                src=filtered_xyz.grad,
                index=this_filter.reshape(-1, 1).expand(-1, 3),
            )
            _opacity_gpu_grad.scatter_add_(
                dim=0, src=_filtered_opacity.grad, index=this_filter.reshape(-1, 1)
            )
            _scaling_gpu_grad.scatter_add_(
                dim=0,
                src=_filtered_scaling.grad,
                index=this_filter.reshape(-1, 1).expand(-1, 3),
            )
            _rotation_gpu_grad.scatter_add_(
                dim=0,
                src=_filtered_rotation.grad,
                index=this_filter.reshape(-1, 1).expand(-1, 4),
            )
            _features_dc_gpu_grad.view(-1, 3).scatter_add_(
                dim=0,
                src=_filtered_features_dc.grad,
                index=this_filter.reshape(-1, 1).expand(-1, 3),
            )
            _features_rest_gpu_grad.view(-1, 45).scatter_add_(
                dim=0,
                src=_filtered_features_rest.grad,
                index=this_filter.reshape(-1, 1).expand(-1, 45),
            )
            torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 3.5: Update visibility mask (for sparse Adam)
        # ------------------------------------------------------------------------
        if sparse_adam:
            torch.cuda.nvtx.range_push("update visibility")
            # Mark which gaussians received gradients in this iteration
            src = torch.ones(
                (len(filters[micro_idx]),), dtype=torch.bool, device="cuda"
            )
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
            args.cpu_adam_trailing_overhead[
                "from_default_stream"
            ] += cpu_adam_trailing_start_event.elapsed_time(cpu_adam_trailing_end_event)

    timers.stop("grad scale + optimizer step + zero grad")
    torch.cuda.synchronize()

    return losses, visibility
