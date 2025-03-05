import torch
import utils.general_utils as utils


def densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        for radii, visibility_filter, screenspace_mean2D in zip(
            batched_screenspace_pkg["batched_locally_preprocessed_radii"],
            batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"],
            batched_screenspace_pkg["batched_locally_preprocessed_mean2D"],
        ):
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(screenspace_mean2D, visibility_filter)
        timers.stop("densification_update_stats")

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            gaussians.densify_and_prune(
                args.densify_grad_threshold,
                args.min_opacity,
                scene.cameras_extent,
                size_threshold,
            )
            timers.stop("densify_and_prune")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write(
                    "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                        iteration,
                        iteration + args.bsz,
                        num_3dgs_before_redistribute,
                        num_3dgs_after_redistribute,
                    )
                )

            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=True
            )

            utils.inc_densify_iter()

        if (
            utils.check_update_at_this_iter(
                iteration, args.bsz, args.opacity_reset_interval, 0
            )
            and iteration + args.bsz <= args.opacity_reset_until_iter
        ):
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )


def gsplat_densification(iteration, scene, gaussians, batched_screenspace_pkg, offload=False, stat_only=False, densify_only=False):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        if not densify_only:
            timers.start("densification_update_stats")
            image_width = batched_screenspace_pkg["image_width"]
            image_height = batched_screenspace_pkg["image_height"]
            send2gpu_filter = batched_screenspace_pkg["send2gpu_filter"]
            batched_screenspace_mean2D_grad = batched_screenspace_pkg[
                "batched_locally_preprocessed_mean2D"
            ].grad
            for i, (radii, visibility_filter) in enumerate(
                zip(
                    batched_screenspace_pkg["batched_locally_preprocessed_radii"],
                    batched_screenspace_pkg[
                        "batched_locally_preprocessed_visibility_filter"
                    ],
                )
            ):
                if args.offload:
                    if args.gpu_cache == "no_cache":
                        (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = send2gpu_filter
                        
                        radii_cpu = radii.cpu() # (len(send2gpu_final_filter_indices), )
                        assert radii.shape[0] == send2gpu_final_filter_indices.shape[0], "radii.shape[0] != send2gpu_final_filter_indices.shape[0]"

                        send2gpu_final_filter_indices_cpu = send2gpu_final_filter_indices.cpu() # (len(send2gpu_final_filter_indices), )
                        batched_grad_cpu = batched_screenspace_mean2D_grad[i].cpu() # (len(send2gpu_final_filter_indices), 3)
                        
                        gaussians.max_radii2D[send2gpu_final_filter_indices_cpu] = torch.max(
                            gaussians.max_radii2D[send2gpu_final_filter_indices_cpu], radii_cpu
                        )

                        gaussians.gsplat_add_densification_stats_exact_filter(
                            batched_grad_cpu,
                            send2gpu_final_filter_indices_cpu,
                            image_width,
                            image_height,
                        )
                        
                    elif args.gpu_cache == "xyzosr":
                        (infrustum_radii_opacities_filter_indices, send2gpu_final_filter_indices) = send2gpu_filter
                        
                        # radii_cpu = radii.cpu() # (len(send2gpu_final_filter_indices), )
                        assert radii.shape[0] == send2gpu_final_filter_indices.shape[0], "radii.shape[0] != send2gpu_final_filter_indices.shape[0]"

                        # send2gpu_final_filter_indices_cpu = send2gpu_final_filter_indices.cpu() # (len(send2gpu_final_filter_indices), )
                        batched_grad = batched_screenspace_mean2D_grad[i] # (len(send2gpu_final_filter_indices), 3)
                        
                        gaussians.max_radii2D[send2gpu_final_filter_indices] = torch.max(
                            gaussians.max_radii2D[send2gpu_final_filter_indices], radii
                        )

                        gaussians.gsplat_add_densification_stats_exact_filter(
                            batched_grad,
                            send2gpu_final_filter_indices,
                            image_width,
                            image_height,
                        )
                        
                    
                    else:
                        raise ValueError("Invalid gpu cache strategy.")
                
                else:
                    batched_grad = batched_screenspace_mean2D_grad[i]
                    send2gpu_visibility_filter = visibility_filter
                    
                    gaussians.max_radii2D[send2gpu_visibility_filter] = torch.max(
                        gaussians.max_radii2D[send2gpu_visibility_filter], radii[visibility_filter]
                    )
                    gaussians.gsplat_add_densification_stats(
                        batched_grad,
                        send2gpu_visibility_filter,
                        visibility_filter,
                        image_width,
                        image_height,
                    )
            timers.stop("densification_update_stats")
        
        if not stat_only:
            if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
                iteration, args.bsz, args.densification_interval, 0
            ):
                assert (
                    args.stop_update_param == False
                ), "stop_update_param must be false for densification; because it is a flag for debugging."
                # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

                gaussians.optimizer.zero_grad(set_to_none=True) # free old tensors' grads before densification

                timers.start("densify_and_prune")
                size_threshold = 20 if iteration > args.opacity_reset_interval else None
                gaussians.densify_and_prune(
                    args.densify_grad_threshold,
                    args.min_opacity,
                    scene.cameras_extent,
                    size_threshold,
                )
                timers.stop("densify_and_prune")
                
                # print(gaussians.parameters_buffer)
                # print(gaussians.parameters_grad_buffer)

                # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
                # if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                #     num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                #     timers.start("redistribute_gaussians")
                #     gaussians.redistribute_gaussians()
                #     timers.stop("redistribute_gaussians")
                #     num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                #     log_file.write(
                #         "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                #             iteration,
                #             iteration + args.bsz,
                #             num_3dgs_before_redistribute,
                #             num_3dgs_after_redistribute,
                #         )
                #     )

                utils.check_memory_usage(
                    log_file, args, iteration, gaussians, before_densification_stop=True
                )

                utils.inc_densify_iter()

            if (
                utils.check_update_at_this_iter(
                    iteration, args.bsz, args.opacity_reset_interval, 0
                )
                and iteration + args.bsz <= args.opacity_reset_until_iter
            ):
                timers.start("reset_opacity")
                gaussians.reset_opacity()
                timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )

def update_densification_stats_pipelineoffload_xyzosr(
    scene,
    gaussians,
    image_height,
    image_width,
    send2gpu_final_filter_indices,
    means2d_grad,
    radii,
):
    iteration = utils.get_cur_iter()
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    assert args.offload
    # assert args.pipelined_offload
    # assert args.gpu_cache == "xyzosr"
    assert radii.shape[0] == send2gpu_final_filter_indices.shape[0], f"radii.shape[0]={radii.shape[0]}, send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}"
    assert send2gpu_final_filter_indices.shape[0] == means2d_grad.shape[0], f"send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}, means2d_grad.shape[0]={means2d_grad.shape[0]}"

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        # timers.start("densification")

        # timers.start("densification_update_stats")
        gaussians.max_radii2D[send2gpu_final_filter_indices] = torch.max(
            gaussians.max_radii2D[send2gpu_final_filter_indices], radii
        )
        gaussians.gsplat_add_densification_stats_exact_filter(
            means2d_grad,
            send2gpu_final_filter_indices,
            image_width,
            image_height,
        )           
        # timers.stop("densification_update_stats")

        # timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )

def update_densification_stats_baseline_accumGrads(
    scene,
    gaussians,
    image_height,
    image_width,
    means2d_grad,
    radii,
    visibility,
):
    iteration = utils.get_cur_iter()
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # assert not args.offload
    # assert radii.shape[0] == send2gpu_final_filter_indices.shape[0], f"radii.shape[0]={radii.shape[0]}, send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}"
    # assert send2gpu_final_filter_indices.shape[0] == means2d_grad.shape[0], f"send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}, means2d_grad.shape[0]={means2d_grad.shape[0]}"

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        # timers.start("densification")

        # timers.start("densification_update_stats")
        if args.packed:
            radii = radii.squeeze(0)

            gaussians.max_radii2D[visibility] = torch.max(gaussians.max_radii2D[visibility], radii)
            gaussians.packed_add_densification_stats(
                means2d_grad,
                visibility,
                image_width,
                image_height,
            )       

        else:
            radii = radii.squeeze(0)
            visibility_filter = radii > 0

            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.gsplat_add_densification_stats(
                means2d_grad.squeeze(0),
                visibility_filter,
                visibility_filter,
                image_width,
                image_height,
            )           
        # timers.stop("densification_update_stats")

        # timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )
