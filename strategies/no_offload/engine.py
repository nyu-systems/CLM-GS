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
from densification import update_densification_stats_baseline_accum_grads
from strategies.base_engine import torch_compiled_loss


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
        degrees_to_use=sh_degree, dirs=dirs, coeffs=shs.unsqueeze(0), masks=(radiis > 0)
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


def baseline_accumGrads_impl(
    gaussians,
    scene,
    batched_cameras,
    background,
    scaling_modifier=1.0,
    sparse_adam=False,
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

    visibility = (
        torch.zeros((means3D.shape[0],), dtype=torch.bool, device="cuda")
        if sparse_adam
        else None
    )

    for micro_idx, camera in enumerate(batched_cameras):
        torch.cuda.nvtx.range_push(f"micro idx {micro_idx}")
        torch.cuda.nvtx.range_push("forward")
        rendered_image, means2D, radiis, gaussian_ids = baseline_accumGrads_micro_step(
            means3D, opacities, scales, rotations, shs, sh_degree, camera, background
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
            update_densification_stats_baseline_accum_grads(
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
