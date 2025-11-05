import torch
import math
import utils.general_utils as utils
from utils.loss_utils import l1_loss
from clm_kernels import fused_ssim
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)

LAMBDA_DSSIM = 0.2  # Loss weight for SSIM
TILE_SIZE = 16


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