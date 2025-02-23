#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import utils.general_utils as utils
import torch.distributed as dist
import cpu_adam
from optimizer import UnifiedAdam, SelectiveAdam
import numba.cuda

lr_scale_fns = {
    "linear": lambda x: x,
    "sqrt": lambda x: np.sqrt(x),
}


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        args = utils.get_args()
        self.args = args
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._parameters = torch.empty(0)
        self.param_dims = torch.empty(0)
        self.param_dims_presum_rshift = torch.empty(0)
        self.col2attr = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(
            0
        )  # TODO: deal with self.send_to_gpui_cnt
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device = "cpu" if args.offload and args.gpu_cache == "no_cache" else "cuda"
        self.mxw_debug = args.mxw_debug

    def capture(self):
        assert false, "optimizer not fixed yet"
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,  # TODO: deal with self.send_to_gpui_cnt
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        assert False, "optimizer not fixed yet"
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        if self._features_rest.dim == 3:
            self._features_rest = self._features_rest.view(self._features_rest.shape[0], -1)
        if self.device == 'cpu' and self.mxw_debug == 'cat':
            self._parameters = torch.cat(
                (self._xyz, self._opacity, self._scaling, self._rotation, self._features_dc, self._features_rest),
                dim=1
            ).pin_memory()
        self.training_setup(training_args)
        self.xyz_gradient_accum = (
            xyz_gradient_accum  # TODO: deal with self.send_to_gpui_cnt
        )
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # TODO: offload: Init gaussian params on cpu.
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, subsample_ratio=1.0):
        log_file = utils.get_log_file()
        # loading could replicated on all ranks.
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = (
            torch.tensor(np.asarray(pcd.points)).float().to("cuda")
        )  # It is not contiguous
        fused_point_cloud = fused_point_cloud.contiguous()  # Now it's contiguous
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to("cuda"))
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .to("cuda")
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        N = fused_point_cloud.shape[0]
        print("Number of points before initialization : ", N)

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).to("cuda")

        if subsample_ratio != 1.0:
            assert subsample_ratio > 0 and subsample_ratio < 1
            sub_N = int(N * subsample_ratio)
            print("Downsample ratio: ", subsample_ratio)
            print("Number of points after downsampling : ", sub_N)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            subsampled_set_gpu, _ = torch.randperm(N)[:sub_N].sort()
            fused_point_cloud = fused_point_cloud[subsampled_set_gpu]
            features = features[subsampled_set_gpu]
            scales = scales[subsampled_set_gpu]
            N = sub_N           

        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (N, 1), dtype=torch.float, device="cuda"
            )
        )

        # The above computation/memory is replicated on all ranks. Because initialization is small, it's ok.
        # Split the point cloud across the ranks.
        args = utils.get_args()
        if (
            args.gaussians_distribution
        ):  # shard 3dgs storage across all GPU including dp and mp groups.
            shard_world_size = utils.DEFAULT_GROUP.size()
            shard_rank = utils.DEFAULT_GROUP.rank()

            point_ind_l, point_ind_r = utils.get_local_chunk_l_r(
                N, shard_world_size, shard_rank
            )
            fused_point_cloud = fused_point_cloud[point_ind_l:point_ind_r].contiguous()
            features = features[point_ind_l:point_ind_r].contiguous()
            scales = scales[point_ind_l:point_ind_r].contiguous()
            rots = rots[point_ind_l:point_ind_r].contiguous()
            opacities = opacities[point_ind_l:point_ind_r].contiguous()
            log_file.write(
                "rank: {}, Number of initialized points: {}\n".format(
                    utils.GLOBAL_RANK, N
                )
            )
            # print("rank", utils.GLOBAL_RANK, "Number of initialized points after gaussians_distribution : ", fused_point_cloud.shape[0])

        if args.drop_initial_3dgs_p > 0.0:
            # drop each point with probability args.drop_initial_3dgs_p
            drop_mask = (
                np.random.rand(N) > args.drop_initial_3dgs_p
            )
            fused_point_cloud = fused_point_cloud[drop_mask]
            features = features[drop_mask]
            scales = scales[drop_mask]
            rots = rots[drop_mask]
            opacities = opacities[drop_mask]
            log_file.write(
                "rank: {}, Number of initialized points after random drop: {}\n".format(
                    utils.GLOBAL_RANK, N
                )
            )
            # print("rank", utils.GLOBAL_RANK, "Number of initialized points after random drop : ", fused_point_cloud.shape[0])
        
        # Init parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.sum_visible_count_in_one_batch = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def create_from_pcd_offloaded(self, pcd: BasicPointCloud, spatial_lr_scale: float, subsample_ratio=1.0):
        log_file = utils.get_log_file()
        self.spatial_lr_scale = spatial_lr_scale
        
        self.parameters_buffer = torch.empty(0)
        self.parameters_grad_buffer = torch.empty(0)
        
        if self.args.gpu_cache == "no_cache":
            self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 59), dtype=torch.float32, pin_memory=True)
            self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 59), dtype=torch.float32, pin_memory=True)
            
            fused_point_cloud = (torch.tensor(np.asarray(pcd.points)).float().to("cpu"))  # It is not contiguous
            fused_point_cloud = fused_point_cloud.contiguous()  # Now it's contiguous
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to("cpu"))
            features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to("cpu"))
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            N = fused_point_cloud.shape[0]
            print("Number of points before initialization : ", N)

            dist2 = torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).to("cpu")
            rots = torch.zeros((N, 4), device="cpu")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cpu"))
            
            features_dc = features[:, :, 0:1].transpose(1, 2).contiguous().view(N, -1) # (N, 1, 3) -> (N, 3)
            features_rest = features[:, :, 1:].transpose(1, 2).contiguous().view(N, -1) # (N, 15, 3) -> (N, 45)
            dims = [fused_point_cloud.shape[1], opacities.shape[1], scales.shape[1], rots.shape[1], features_dc.shape[1], features_rest.shape[1]]
            # parameters = torch.empty((N, sum(dims)), pin_memory=True)          
            torch.cat((fused_point_cloud, opacities, scales, rots, features_dc, features_rest), dim=1, out=self.parameters_buffer[:N])
            
            self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
            self._xyz, self._opacity, self._scaling, self._rotation, self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1) 
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cpu")
            self.sum_visible_count_in_one_batch = torch.zeros((self.get_xyz.shape[0]), device="cpu")
            
            self.param_dims = torch.tensor(dims, dtype=torch.int, device='cuda')
            self.param_dims_presum_rshift = torch.cumsum(self.param_dims, dtype=torch.int, dim=0) - self.param_dims
            self.col2attr = torch.empty((sum(dims),), dtype=torch.int, device='cuda')
            for i in range(sum(dims)):
                if i < self.param_dims_presum_rshift[1]:
                    self.col2attr[i] = 0
                elif i < self.param_dims_presum_rshift[2]:
                    self.col2attr[i] = 1
                elif i < self.param_dims_presum_rshift[3]:
                    self.col2attr[i] = 2
                elif i < self.param_dims_presum_rshift[4]:
                    self.col2attr[i] = 3
                elif i < self.param_dims_presum_rshift[5]:
                    self.col2attr[i] = 4
                else:
                    self.col2attr[i] = 5
        
        elif self.args.gpu_cache == "xyzosr":
            # HACK: pytorch cachingHostAllocator automatically rounds up and does not expose a free-cache api.
            # To avoid wasted system ram, we convert from numba pinned array as a workaround.
            # (however, numba never frees the pinned memory as well, so we only allocate a big enough buffer once)

            # self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 48), dtype=torch.float32, pin_memory=True)
            # self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 48), dtype=torch.float32, pin_memory=True)
            parameters_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
            self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
            parameters_grad_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
            self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
            assert self.parameters_buffer.is_pinned()
            assert self.parameters_grad_buffer.is_pinned()

            fused_point_cloud = (torch.tensor(np.asarray(pcd.points)).float().to("cuda"))  # It is not contiguous
            fused_point_cloud = fused_point_cloud.contiguous()  # Now it's contiguous
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to("cpu"))
            features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to("cpu"))
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            N = fused_point_cloud.shape[0]
            print("Number of points before initialization : ", N)

            dist2 = torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).to("cuda")

            if subsample_ratio != 1.0:
                assert subsample_ratio > 0 and subsample_ratio < 1
                sub_N = int(N * subsample_ratio)
                print("Downsample ratio: ", subsample_ratio)
                print("Number of points after downsampling : ", sub_N)

                perm_generator = torch.Generator()
                perm_generator.manual_seed(1)
                subsampled_set_gpu, _ = torch.randperm(N)[:sub_N].sort()
                subsampled_set_cpu = subsampled_set_gpu.to("cpu")
                fused_point_cloud = fused_point_cloud[subsampled_set_gpu]
                features = features[subsampled_set_cpu]
                scales = scales[subsampled_set_gpu]
                N = sub_N           

            rots = torch.zeros((N, 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))
            
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            
            features_dc = features[:, :, 0:1].transpose(1, 2).contiguous().view(N, -1) # (N, 1, 3) -> (N, 3)
            features_rest = features[:, :, 1:].transpose(1, 2).contiguous().view(N, -1) # (N, 15, 3) -> (N, 45)
            dims = [features_dc.shape[1], features_rest.shape[1]]
            torch.cat((features_dc, features_rest), dim=1, out=self.parameters_buffer[:N])
            
            self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
            self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1) 
            self.max_radii2D = torch.zeros((N), device="cuda")
            self.sum_visible_count_in_one_batch = torch.zeros((N), device="cuda")
            
            self.param_dims = torch.tensor(dims, dtype=torch.int, device='cuda')
            # self.param_dims_presum_rshift = torch.cumsum(self.param_dims, dtype=torch.int, dim=0) - self.param_dims
            # self.col2attr = torch.empty((sum(dims),), dtype=torch.int, device='cuda')
            # for i in range(sum(dims)):
            #     if i < self.param_dims_presum_rshift[1]:
            #         self.col2attr[i] = 0 # features_dc
            #     else:
            #         self.col2attr[i] = 1 # features_rest
        
        else:
            raise ValueError("Gpu cache strategy undefined")

    def all_parameters(self):
        if self.args.offload:
            if self.args.gpu_cache == "no_cache":
                return [self._parameters]
            elif self.args.gpu_cache == "xyzosr":
                return [
                    self._xyz,
                    self._opacity,
                    self._scaling,
                    self._rotation,
                    self._parameters,
                ]
            else:
                raise Exception("Gpu cache strategy undefined")
        else:
            return [
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
            ]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device) 

        shard_world_size = self.group_for_redistribution().size()
        self.send_to_gpui_cnt = torch.zeros(
            (self.get_xyz.shape[0], shard_world_size), dtype=torch.int, device=self.device
        ) # TODO: Check where this param is called. If it's used for gpu, should stil init it there.

        args = utils.get_args()
        log_file = utils.get_log_file()
        
        # Setup the optimizer
        if args.offload:
            if args.gpu_cache == "no_cache":
                l = [
                    {
                        "params": [self._parameters],
                        "lr": training_args.position_lr_init
                        * self.spatial_lr_scale
                        * args.lr_scale_pos_and_scale,
                        "name": "parameters"
                    }
                ]
                column_sizes = [3, 1, 3, 4, 3, 45]
                # total_columns = sum(column_sizes)
                column_lrs = [
                    training_args.position_lr_init * self.spatial_lr_scale * args.lr_scale_pos_and_scale,
                    training_args.opacity_lr,
                    training_args.scaling_lr * args.lr_scale_pos_and_scale,
                    training_args.rotation_lr,
                    training_args.feature_lr,
                    training_args.feature_lr / 20.0]
                
                self.optimizer = cpu_adam.FusedCPUAdam(
                    l,
                    columns_sizes=column_sizes,
                    columns_lr=column_lrs,
                    lr=0.0,
                    bias_correction=True, # This True is required. 
                    betas=(0.9, 0.999),
                    eps=1e-15,
                    weight_decay=0,
                    amsgrad=False,
                    adamw_mode=False,
                    fp32_optimizer_states=True
                )
                
            elif args.gpu_cache == "xyzosr":
                l = [
                    {
                        "params": [self._xyz],
                        "lr": training_args.position_lr_init
                        * self.spatial_lr_scale
                        * args.lr_scale_pos_and_scale,
                        "name": "xyz",
                    },
                    {
                        "params": [self._opacity],
                        "lr": training_args.opacity_lr,
                        "name": "opacity",
                    },
                    {
                        "params": [self._scaling],
                        "lr": training_args.scaling_lr * args.lr_scale_pos_and_scale,
                        "name": "scaling",
                    },
                    {
                        "params": [self._rotation],
                        "lr": training_args.rotation_lr,
                        "name": "rotation",
                    },
                    {
                        "params": [self._parameters], # concatenated shs
                        "lr": training_args.feature_lr,
                        "name": "parameters"
                    },
                ]
                column_sizes = [3, 45]
                column_lrs = [
                    training_args.feature_lr,
                    training_args.feature_lr / 20.0]
                
                self.optimizer = UnifiedAdam(
                    l,
                    column_sizes,
                    column_lrs,
                    lr=0.0,
                    bias_correction=True, # This True is required. 
                    betas=(0.9, 0.999),
                    eps=1e-15,
                    weight_decay=0,
                    amsgrad=False,
                    adamw_mode=False,
                    fp32_optimizer_states=True,
                    fused=True if args.fused_adam == "torch_fused" else False
                )
                
            else:
                raise ValueError("Invalid gpu cache strategy.")
            
        else:
            l = [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init
                    * self.spatial_lr_scale
                    * args.lr_scale_pos_and_scale,
                    "name": "xyz",
                },
                {
                    "params": [self._features_dc],
                    "lr": training_args.feature_lr,
                    "name": "f_dc",
                },
                {
                    "params": [self._features_rest],
                    "lr": training_args.feature_lr / 20.0,
                    "name": "f_rest",
                },
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
                },
                {
                    "params": [self._scaling],
                    "lr": training_args.scaling_lr * args.lr_scale_pos_and_scale,
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                },
            ]
            if args.sparse_adam:
                self.optimizer = SelectiveAdam(l, lr=0.0, eps=1e-15)
            else:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, fused=True if args.fused_adam == "torch_fused" else False)

        # Scale learning rates according to bsz.
        bsz = args.bsz
        for param_group in self.optimizer.param_groups:
            if training_args.lr_scale_mode == "linear":
                lr_scale = bsz
                param_group["lr"] *= lr_scale
            elif training_args.lr_scale_mode == "sqrt":
                lr_scale = np.sqrt(bsz)
                param_group["lr"] *= lr_scale
                if "eps" in param_group:  # Adam
                    param_group["eps"] /= lr_scale
                    param_group["betas"] = [beta**bsz for beta in param_group["betas"]]
                    # utils.print_rank_0(param_group["name"] + " betas: " + str(param_group["betas"]))
                    log_file.write(
                        param_group["name"]
                        + " betas: "
                        + str(param_group["betas"])
                        + "\n"
                    )
            elif training_args.lr_scale_mode == "accumu":
                lr_scale = 1
            else:
                assert (
                    False
                ), f"lr_scale_mode {training_args.lr_scale_mode} not supported."
        
        if args.offload:
            if training_args.lr_scale_mode == "linear":
                lr_scale = bsz
                self.optimizer.columns_lr *= lr_scale
            elif training_args.lr_scale_mode == "sqrt":
                lr_scale = np.sqrt(bsz)
                self.optimizer.columns_lr *= lr_scale
            

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_final=training_args.position_lr_final
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        utils.check_initial_gpu_memory_usage("after training_setup")

    def log_gaussian_stats(self):
        # log the statistics of the gaussian model
        # number of total 3dgs on this rank
        num_3dgs = self._xyz.shape[0]
        # average size of 3dgs
        avg_size = torch.mean(torch.max(self.get_scaling, dim=1).values).item()
        # average opacity
        avg_opacity = torch.mean(self.get_opacity).item()
        stats = {
            "num_3dgs": num_3dgs,
            "avg_size": avg_size,
            "avg_opacity": avg_opacity,
        }

        # get the exp_avg, exp_avg_sq state for all parameters
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    exp_avg_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg"], dim=-1)
                    ).item()
                    exp_avg_sq_dict[group["name"]] = torch.mean(
                        torch.norm(stored_state["exp_avg_sq"], dim=-1)
                    ).item()
        return stats, exp_avg_dict, exp_avg_sq_dict

    def sync_gradients_for_replicated_3dgs_storage(self, batched_screenspace_pkg):
        args = utils.get_args()

        if "visible_count" in args.grad_normalization_mode:
            # allgather visibility filder from all dp workers, so that each worker contains the visibility filter of all data points.
            batched_locally_preprocessed_visibility_filter_int = [
                x.int()
                for x in batched_screenspace_pkg[
                    "batched_locally_preprocessed_visibility_filter"
                ]
            ]
            sum_batched_locally_preprocessed_visibility_filter_int = torch.sum(
                torch.stack(batched_locally_preprocessed_visibility_filter_int), dim=0
            )
            batched_screenspace_pkg[
                "sum_batched_locally_preprocessed_visibility_filter_int"
            ] = sum_batched_locally_preprocessed_visibility_filter_int

        if args.sync_grad_mode == "dense":
            sync_func = sync_gradients_densely
        elif args.sync_grad_mode == "sparse":
            sync_func = sync_gradients_sparsely
        elif args.sync_grad_mode == "fused_dense":
            sync_func = sync_gradients_fused_densely
        elif args.sync_grad_mode == "fused_sparse":
            sync_func = sync_gradients_fused_sparsely
        else:
            assert False, f"sync_grad_mode {args.sync_grad_mode} not supported."

        if not args.gaussians_distribution and utils.DEFAULT_GROUP.size() > 1:
            sync_func(self, utils.DEFAULT_GROUP)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if self.args.offload and self.args.gpu_cache == "no_cache":
                # `xyz` is concatenated in `parameters`
                if param_group["name"] == "parameters":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group["lr"] = lr
                    return lr
            else:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group["lr"] = lr
                    return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        features_dc_elems = self._features_dc.shape[1] * self._features_dc.shape[2] if len(self._features_dc.shape) == 3 else self._features_dc.shape[1]
        features_rest_elems = self._features_rest.shape[1] * self._features_rest.shape[2] if len(self._features_rest.shape) == 3 else self._features_rest.shape[1]
        for i in range(features_dc_elems):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest_elems):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l
    
    def save_tensors(
        self, parent_path
    ):
        args = utils.get_args()
        assert args.offload, "Only implemented for offloading."

        mkdir_p(parent_path)

        torch.save(self._xyz, os.path.join(parent_path, "xyz.pt"))
        torch.save(self._opacity, os.path.join(parent_path, "opacity.pt"))
        torch.save(self._scaling, os.path.join(parent_path, "scaling.pt"))
        torch.save(self._rotation, os.path.join(parent_path, "rotation.pt"))
        torch.save(self._parameters, os.path.join(parent_path, "parameters.pt"))
        
    def save_sub_plys(
        self, path, n_split, split_size
    ):
        args = utils.get_args()
        assert args.offload, "Only implemented for offloading."
        _xyz = _features_dc = _features_rest = _opacity = _scaling = _rotation = None
        utils.log_cpu_memory_usage("start save_ply")
        _xyz = self._xyz
        _features_dc = self._features_dc
        _features_rest = self._features_rest
        _opacity = self._opacity
        _scaling = self._scaling
        _rotation = self._rotation

        for i in range(n_split):
            assert path.endswith(".ply")
            this_path = (
                path[:-4]
                + "_rk"
                + str(i)
                + "_ws"
                + str(n_split)
                + ".ply"
            )
            mkdir_p(os.path.dirname(this_path))

            start = i * split_size
            end = min((i + 1) * split_size, _xyz.shape[0])

            xyz = _xyz.detach()[start:end].cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = (
                _features_dc.detach()[start:end]
                .contiguous()
                .view(-1, 1, 3)
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            f_rest = (
                _features_rest.detach()[start:end]
                .contiguous()
                .view(-1, 15, 3)
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            opacities = _opacity.detach()[start:end].cpu().numpy()
            scale = _scaling.detach()[start:end].cpu().numpy()
            rotation = _rotation.detach()[start:end].cpu().numpy()

            utils.log_cpu_memory_usage(f"[{i/n_split}] after change gpu tensor to cpu numpy")

            dtype_full = [
                (attribute, "f4") for attribute in self.construct_list_of_attributes()
            ]

            elements = np.empty(split_size, dtype=dtype_full)
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
            del xyz, normals, f_dc, f_rest, opacities, scale, rotation
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")

            utils.log_cpu_memory_usage(
                f"[{i/n_split}] after change numpy to plyelement before writing ply file"
            )
            PlyData([el]).write(this_path)
               
        utils.log_cpu_memory_usage("finish write ply file")
        # remark: max_radii2D, xyz_gradient_accum and denom are not saved here; they are save elsewhere.

    def save_ply(
        self, path
    ):  # here, we should be in torch.no_grad() context. train.py ensures that.
        args = utils.get_args()
        _xyz = _features_dc = _features_rest = _opacity = _scaling = _rotation = None
        utils.log_cpu_memory_usage("start save_ply")
        group = utils.DEFAULT_GROUP
        if args.gaussians_distribution and not args.distributed_save:
            # gather all gaussians at rank 0
            def gather_uneven_tensors(tensor):
                # gather size of tensors on different ranks
                tensor_sizes = torch.zeros(
                    (group.size()), dtype=torch.int, device="cuda"
                )
                tensor_sizes[group.rank()] = tensor.shape[0]
                dist.all_reduce(tensor_sizes, op=dist.ReduceOp.SUM)
                # move tensor_sizes to CPU and convert to int list
                tensor_sizes = tensor_sizes.cpu().numpy().tolist()

                # NOTE: Internal implementation of gather could not gather tensors of different sizes.
                # So, I do not use dist.gather(tensor, dst=0) but use dist.send(tensor, dst=0) and dist.recv(tensor, src=i) instead.

                # gather tensors on different ranks using grouped send/recv
                gathered_tensors = []
                if group.rank() == 0:
                    for i in range(group.size()):
                        if i == group.rank():
                            gathered_tensors.append(tensor)
                        else:
                            tensor_from_rk_i = torch.zeros(
                                (tensor_sizes[i],) + tensor.shape[1:],
                                dtype=tensor.dtype,
                                device="cuda",
                            )
                            dist.recv(tensor_from_rk_i, src=i)
                            gathered_tensors.append(tensor_from_rk_i)
                    gathered_tensors = torch.cat(gathered_tensors, dim=0)
                else:
                    dist.send(tensor, dst=0)
                # concatenate gathered tensors

                return (
                    gathered_tensors if group.rank() == 0 else None
                )  # only return gather tensors at rank 0

            _xyz = gather_uneven_tensors(self._xyz)
            _features_dc = gather_uneven_tensors(self._features_dc)
            _features_rest = gather_uneven_tensors(self._features_rest)
            _opacity = gather_uneven_tensors(self._opacity)
            _scaling = gather_uneven_tensors(self._scaling)
            _rotation = gather_uneven_tensors(self._rotation)

            if group.rank() != 0:
                return

        elif args.gaussians_distribution and args.distributed_save:
            assert (
                utils.DEFAULT_GROUP.size() > 1
            ), "distributed_save should be used with more than 1 rank."
            _xyz = self._xyz
            _features_dc = self._features_dc
            _features_rest = self._features_rest
            _opacity = self._opacity
            _scaling = self._scaling
            _rotation = self._rotation
            if path.endswith(".ply"):
                path = (
                    path[:-4]
                    + "_rk"
                    + str(utils.GLOBAL_RANK)
                    + "_ws"
                    + str(utils.WORLD_SIZE)
                    + ".ply"
                )
        elif not args.gaussians_distribution:
            if group.rank() != 0:
                return
            _xyz = self._xyz
            _features_dc = self._features_dc
            _features_rest = self._features_rest
            _opacity = self._opacity
            _scaling = self._scaling
            _rotation = self._rotation
            if path.endswith(".ply"):
                path = (
                    path[:-4]
                    + "_rk"
                    + str(utils.GLOBAL_RANK)
                    + "_ws"
                    + str(utils.WORLD_SIZE)
                    + ".ply"
                )

        mkdir_p(os.path.dirname(path))

        xyz = _xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            _features_dc.detach()
            .contiguous()
            .view(-1, 1, 3)
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            _features_rest.detach()
            .contiguous()
            .view(-1, 15, 3)
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = _opacity.detach().cpu().numpy()
        scale = _scaling.detach().cpu().numpy()
        rotation = _rotation.detach().cpu().numpy()

        utils.log_cpu_memory_usage("after change gpu tensor to cpu numpy")

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        utils.log_cpu_memory_usage(
            "after change numpy to plyelement before writing ply file"
        )
        PlyData([el]).write(path)
        utils.log_cpu_memory_usage("finish write ply file")
        # remark: max_radii2D, xyz_gradient_accum and denom are not saved here; they are save elsewhere.

    def reset_opacity(self):
        utils.LOG_FILE.write("Resetting opacity to 0.01\n")
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        if self.args.offload:
            if self.args.gpu_cache == "no_cache":
                self._parameters[:, 3:4] = opacities_new
                self.cat_replace_opacities_to_optimizer(opacities_new, "parameters")
                self._opacity = self._parameters[:, 3:4]
            elif self.args.gpu_cache == "xyzosr":
                optimizable_tensors = self.replace_tensor_to_unified_adam(opacities_new, "opacity")
                self._opacity = optimizable_tensors["opacity"]
            else:
                raise ValueError("Invalid gpu cache strategy.")
        else:
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def prune_based_on_opacity(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        utils.LOG_FILE.write(
            "Pruning based on opacity. Percent: {:.2f}\n".format(
                100 * prune_mask.sum().item() / prune_mask.shape[0]
            )
        )
        self.prune_points(prune_mask)

    def distributed_load_ply(self, folder):
        # count the number of files like "point_cloud_rk0_ws4.ply"
        world_size = -1
        for f in os.listdir(folder):
            if "_ws" in f:
                world_size = int(f.split("_ws")[1].split(".")[0])
                break
        assert world_size > 0, "world_size should be greater than 1."

        catted_xyz = []
        catted_features_dc = []
        catted_features_rest = []
        catted_opacity = []
        catted_scaling = []
        catted_rotation = []
        for rk in range(min(world_size, self.args.load_ply_max)):
            one_checkpoint_path = (
                folder + "/point_cloud_rk" + str(rk) + "_ws" + str(world_size) + ".ply"
            )
            xyz, features_dc, features_extra, opacities, scales, rots = (
                self.load_raw_ply(one_checkpoint_path)
            )
            catted_xyz.append(xyz)
            catted_features_dc.append(features_dc)
            catted_features_rest.append(features_extra)
            catted_opacity.append(opacities)
            catted_scaling.append(scales)
            catted_rotation.append(rots)
        catted_xyz = np.concatenate(catted_xyz, axis=0)
        catted_features_dc = np.concatenate(catted_features_dc, axis=0)
        catted_features_rest = np.concatenate(catted_features_rest, axis=0)
        catted_opacity = np.concatenate(catted_opacity, axis=0)
        catted_scaling = np.concatenate(catted_scaling, axis=0)
        catted_rotation = np.concatenate(catted_rotation, axis=0)
        
        if self.args.offload:
            self.parameters_buffer = torch.empty(0)
            self.parameters_grad_buffer = torch.zeros(0)
        
            N = catted_xyz.shape[0]
            if catted_features_dc.ndim == 3:
                catted_features_dc = np.transpose(catted_features_dc, (0, 2, 1)).reshape(N, -1)
            if catted_features_rest.ndim == 3:
                catted_features_rest = np.transpose(catted_features_rest, (0, 2, 1)).reshape(N, -1)
            
            if self.args.gpu_cache == "no_cache":
                self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 59), dtype=torch.float, pin_memory=True)
                self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 59), dtype=torch.float, pin_memory=True)
                self.parameters_buffer[:N] = torch.tensor(np.concatenate((catted_xyz, catted_opacity, catted_scaling, catted_rotation, catted_features_dc, catted_features_rest), axis=1))
                self._parameters = nn.Parameter(
                    self.parameters_buffer[:N].requires_grad_(True)
                )
                dims = [catted_xyz.shape[1], catted_opacity.shape[1], catted_scaling.shape[1], catted_rotation.shape[1], catted_features_dc.shape[1], catted_features_rest.shape[1]]
                self._xyz, self._opacity, self._scaling, self._rotation, self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1)
                self.param_dims = torch.tensor(dims, dtype=torch.int, device='cuda')
                self.param_dims_presum_rshift = torch.cumsum(self.param_dims, dtype=torch.int, dim=0) - self.param_dims
                self.col2attr = torch.empty((sum(dims),), dtype=torch.int, device='cuda')
                for i in range(sum(dims)):
                    if i < self.param_dims_presum_rshift[1]:
                        self.col2attr[i] = 0
                    elif i < self.param_dims_presum_rshift[2]:
                        self.col2attr[i] = 1
                    elif i < self.param_dims_presum_rshift[3]:
                        self.col2attr[i] = 2
                    elif i < self.param_dims_presum_rshift[4]:
                        self.col2attr[i] = 3
                    elif i < self.param_dims_presum_rshift[5]:
                        self.col2attr[i] = 4
                    else:
                        self.col2attr[i] = 5
                
                self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cpu")
                self.sum_visible_count_in_one_batch = torch.zeros(
                    (self.get_xyz.shape[0]), device="cpu"
                )
                
            elif self.args.gpu_cache == "xyzosr":
                self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)
                self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)
                
                self.parameters_buffer[:N] = torch.tensor(np.concatenate((catted_features_dc, catted_features_rest), axis=1))
                self._parameters = nn.Parameter(
                    self.parameters_buffer[:N].requires_grad_(True)
                )
                
                self._xyz = nn.Parameter(
                    torch.tensor(
                        catted_xyz, dtype=torch.float, device="cuda"
                    ).requires_grad_(True)
                )
                self._opacity = nn.Parameter(
                    torch.tensor(
                        catted_opacity, dtype=torch.float, device="cuda"
                    ).requires_grad_(True)
                )
                self._scaling = nn.Parameter(
                    torch.tensor(
                        catted_scaling, dtype=torch.float, device="cuda"
                    ).requires_grad_(True)
                )
                self._rotation = nn.Parameter(
                    torch.tensor(
                        catted_rotation, dtype=torch.float, device="cuda"
                    ).requires_grad_(True)
                )
                self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
                self.sum_visible_count_in_one_batch = torch.zeros(
                    (self.get_xyz.shape[0]), device="cuda"
                )
            
            else:
                raise ValueError("Invalid gpu cache strategy.")
            
        else:
            self._xyz = nn.Parameter(
                torch.tensor(catted_xyz, dtype=torch.float, device="cuda").requires_grad_(
                    True
                )
            )
            self._features_dc = nn.Parameter(
                torch.tensor(catted_features_dc, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.tensor(catted_features_rest, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                torch.tensor(
                    catted_opacity, dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                torch.tensor(
                    catted_scaling, dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.tensor(
                    catted_rotation, dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def load_raw_ply(self, path):
        print("Loading ", path)
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        args = utils.get_args()
        # The above computation/memory is replicated on all ranks. Because initialization is small, it's ok.
        # Split the point cloud across the ranks.

        if args.gaussians_distribution and utils.WORLD_SIZE > 1:
            chunk = xyz.shape[0] // utils.WORLD_SIZE + 1
            point_ind_l = chunk * utils.LOCAL_RANK
            point_ind_r = min(chunk * (utils.LOCAL_RANK + 1), xyz.shape[0])
            # xyz = xyz[point_ind_l:point_ind_r].contiguous()
            # features_dc = features_dc[point_ind_l:point_ind_r].contiguous()
            # features_extra = features_extra[point_ind_l:point_ind_r].contiguous()
            # scales = scales[point_ind_l:point_ind_r].contiguous()
            # rots = rots[point_ind_l:point_ind_r].contiguous()
            # opacities = opacities[point_ind_l:point_ind_r].contiguous()

            xyz = np.ascontiguousarray(xyz[point_ind_l:point_ind_r])
            features_dc = np.ascontiguousarray(features_dc[point_ind_l:point_ind_r])
            features_extra = np.ascontiguousarray(
                features_extra[point_ind_l:point_ind_r]
            )
            scales = np.ascontiguousarray(scales[point_ind_l:point_ind_r])
            rots = np.ascontiguousarray(rots[point_ind_l:point_ind_r])
            opacities = np.ascontiguousarray(opacities[point_ind_l:point_ind_r])

        if args.drop_initial_3dgs_p > 0.0:
            # drop each point with probability args.drop_initial_3dgs_p
            drop_mask = np.random.rand(xyz.shape[0]) > args.drop_initial_3dgs_p
            xyz = xyz[drop_mask]
            features_dc = features_dc[drop_mask]
            features_extra = features_extra[drop_mask]
            scales = scales[drop_mask]
            rots = rots[drop_mask]
            opacities = opacities[drop_mask]

        return xyz, features_dc, features_extra, opacities, scales, rots

    def one_file_load_ply(self, folder):
        path = os.path.join(folder, "point_cloud.ply")
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_raw_ply(
            path
        )

        if self.args.offload:
            self.parameters_buffer = torch.empty(0)
            self.parameters_grad_buffer = torch.zeros(0)

            N = xyz.shape[0]
            if features_dc.ndim == 3:
                features_dc = np.transpose(features_dc, (0, 2, 1)).reshape(N, -1)
            if features_extra.ndim == 3:
                features_extra = np.transpose(features_extra, (0, 2, 1)).reshape(N, -1)

            if self.args.gpu_cache == "xyzosr":
                self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)
                self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)

                self.parameters_buffer[:N] = torch.tensor(np.concatenate((features_dc, features_extra), axis=1))
                self._parameters = nn.Parameter(
                    self.parameters_buffer[:N].requires_grad_(True)
                )

                self._xyz = nn.Parameter(
                    torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True)
                )
                self._opacity = nn.Parameter(
                    torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(
                        True
                    )
                )
                self._scaling = nn.Parameter(
                    torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True)
                )
                self._rotation = nn.Parameter(
                    torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True)
                )

                self.active_sh_degree = self.max_sh_degree
            
            else:
                raise ValueError("Not implemented yet.")
        
        else:
            self._xyz = nn.Parameter(
                torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True)
            )
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(
                    True
                )
            )
            self._scaling = nn.Parameter(
                torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True)
            )

            self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path):
        if os.path.exists(os.path.join(path, "point_cloud.ply")):
            self.one_file_load_ply(path)
        else:
            self.distributed_load_ply(path)

    def cat_replace_opacities_to_optimizer(self, opacities_new, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    if "exp_avg" not in stored_state:
                        stored_state["momentum_buffer"][:, 3:4].zero_()
                    else:
                        stored_state["exp_avg"][:, 3:4].zero_()
                        stored_state["exp_avg_sq"][:, 3:4].zero_()
                    
                    self.optimizer.state[group["params"][0]] = stored_state
                    
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    if "exp_avg" not in stored_state:
                        stored_state["momentum_buffer"] = torch.zeros_like(tensor)
                    else:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    
                    del self.optimizer.state[group["params"][0]]
                    
                    if self.device == 'cpu' and (self.mxw_debug == 'fused' or self.mxw_debug == 'cat'):
                        group["params"][0] = nn.Parameter(tensor.pin_memory().requires_grad_(True))
                    else:
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    if self.device == 'cpu' and (self.mxw_debug == 'fused' or self.mxw_debug == 'cat'):
                        group["params"][0] = nn.Parameter(tensor.pin_memory().requires_grad_(True))
                    else:
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def replace_tensor_to_unified_adam(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                assert group["params"][0].is_cuda, "Not implemented for parameters on cpu yet."
                stored_state = self.optimizer.gpu_adam.state.get(group["params"][0], None)
                assert stored_state is not None, "UnifiedAdam is a stateful optimizer."
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.zeros_like(tensor)
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                
                del self.optimizer.gpu_adam.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            mask = mask.to(group["params"][0].device.type)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][mask]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                
                if group["params"][0].is_cuda:
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                else:
                    assert mask.dim() == 1
                    self.parameters_buffer[:torch.sum(mask)] = group["params"][0][mask]
                    group["params"][0] = nn.Parameter(
                        (self.parameters_buffer[:torch.sum(mask)].requires_grad_(True))
                    )

                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]    
            else:
                if group["params"][0].is_cuda:
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                else:
                    assert mask.dim() == 1
                    self.parameters_buffer[:torch.sum(mask)] = group["params"][0][mask]
                    group["params"][0] = nn.Parameter(
                        (self.parameters_buffer[:torch.sum(mask)].requires_grad_(True))
                    )
                optimizable_tensors[group["name"]] = group["params"][0]
                
        return optimizable_tensors
    
    def _prune_unified_adam(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["params"][0].is_cuda:
                stored_state = self.optimizer.gpu_adam.state.get(group["params"][0], None)
                mask = mask.to(group["params"][0].device.type)
                assert stored_state is not None, "Unified adam is a stateful optimizer."

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][mask]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.gpu_adam.state[group["params"][0]]

                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )

                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            
            else:
                stored_state = self.optimizer.cpu_adam.state.get(group["params"][0], None)
                mask = mask.to(group["params"][0].device.type)
                assert stored_state is not None, "Unified adam is a stateful optimizer."

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][mask]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.cpu_adam.state[group["params"][0]]

                assert mask.dim() == 1
                self.parameters_buffer[:torch.sum(mask)] = group["params"][0][mask]
                group["params"][0] = nn.Parameter(
                    (self.parameters_buffer[:torch.sum(mask)].requires_grad_(True))
                )

                self.optimizer.cpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        self.optimizer.state = self.optimizer.gpu_adam.state | self.optimizer.cpu_adam.state   
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self.args.offload and self.args.gpu_cache == "xyzosr":
            optimizable_tensors = self._prune_unified_adam(valid_points_mask)
        else:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        if self.args.offload:
            if self.args.gpu_cache == "no_cache":
                self._parameters = optimizable_tensors["parameters"]
                dims = [self._xyz.shape[1], self._opacity.shape[1], self._scaling.shape[1], self._rotation.shape[1], self._features_dc.shape[1], self._features_rest.shape[1]]           
                self._xyz, self._opacity, self._scaling, self._rotation, self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1)
                
                assert self._parameters.is_pinned()
                assert self._xyz.is_pinned()
                assert self._opacity.is_pinned()
                assert self._scaling.is_pinned()
                assert self._rotation.is_pinned()
                assert self._features_dc.is_pinned()
                assert self._features_rest.is_pinned()
                
            elif self.args.gpu_cache == "xyzosr":
                self._xyz = optimizable_tensors["xyz"]
                self._opacity = optimizable_tensors["opacity"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._parameters = optimizable_tensors["parameters"]
                dims = [3, 45]           
                self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1)
                
                assert self._xyz.is_cuda
                assert self._opacity.is_cuda
                assert self._scaling.is_cuda
                assert self._rotation.is_cuda
                assert self._parameters.is_pinned()
                assert self._features_dc.is_pinned()
                assert self._features_rest.is_pinned()
            
            else:
                raise ValueError("Invalid gpu cache strategy.")
            
        else:
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.send_to_gpui_cnt = self.send_to_gpui_cnt[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.sum_visible_count_in_one_batch = self.sum_visible_count_in_one_batch[
            valid_points_mask
        ]
        
        
        if self.device == 'cpu' and self.mxw_debug == 'fused':
            assert self._xyz.is_pinned(), "`[after prunning] self._xyz` is not in pinned memory"
            assert self._scaling.is_pinned(), "`[after prunning] self._scaling` is not in pinned memory"
            assert self._rotation.is_pinned(), "`[after prunning] self._rotation` is not in pinned memory"
            assert self._opacity.is_pinned(), "`[after prunning] self._opacity` is not in pinned memory"
            assert self._features_dc.is_pinned(), "`[after prunning] self._features_dc` is not in pinned memory"
            assert self._features_rest.is_pinned(), "`[after prunning] self._features_rest` is not in pinned memory"
        elif self.device == 'cpu' and self.mxw_debug == 'cat':
            assert self._parameters.is_pinned(), "`[after prunning] self._parameters` is not in pinned memory"


        if self.device == 'cpu' and self.mxw_debug == 'fused':
            assert self._xyz.is_pinned(), "`[after prunning] self._xyz` is not in pinned memory"
            assert self._scaling.is_pinned(), "`[after prunning] self._scaling` is not in pinned memory"
            assert self._rotation.is_pinned(), "`[after prunning] self._rotation` is not in pinned memory"
            assert self._opacity.is_pinned(), "`[after prunning] self._opacity` is not in pinned memory"
            assert self._features_dc.is_pinned(), "`[after prunning] self._features_dc` is not in pinned memory"
            assert self._features_rest.is_pinned(), "`[after prunning] self._features_rest` is not in pinned memory"
        elif self.device == 'cpu' and self.mxw_debug == 'cat':
            assert self._parameters.is_pinned(), "`[after prunning] self._parameters` is not in pinned memory"

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if tensors_dict[group["name"]] is None:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (
                            stored_state["momentum_buffer"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                del self.optimizer.state[group["params"][0]]
                
                # Update parameters.
                if group["params"][0].is_cuda:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                else:
                    N = group["params"][0].shape[0]
                    N_ext = extension_tensor.shape[0]
                    self.parameters_buffer[N:(N + N_ext)] = extension_tensor
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:(N + N_ext)].requires_grad_(True)
                    )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["params"][0].is_cuda:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                else:
                    N = group["params"][0].shape[0]
                    N_ext = extension_tensor.shape[0]
                    self.parameters_buffer[N:(N + N_ext)] = extension_tensor
                    group["params"][0] = nn.Parameter(
                        self.parameters_buffer[:(N + N_ext)].requires_grad_(True)
                    )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

    def cat_tensors_to_unified_adam(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if tensors_dict[group["name"]] is None:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]

            # stored_state = self.optimizer.state.get(group["params"][0], None)
            if group["params"][0].is_cuda:
                stored_state = self.optimizer.gpu_adam.state.get(group["params"][0], None)

                assert stored_state is not None, "Unified adam is a stateful optimizer."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (
                            stored_state["momentum_buffer"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                del self.optimizer.gpu_adam.state[group["params"][0]]
                
                # Update parameters.
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            
            else:
                stored_state = self.optimizer.cpu_adam.state.get(group["params"][0], None)

                assert stored_state is not None, "Unified adam is a stateful optimizer."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (
                            stored_state["momentum_buffer"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                del self.optimizer.cpu_adam.state[group["params"][0]]
                
                # Update parameters.
                N = group["params"][0].shape[0]
                N_ext = extension_tensor.shape[0]
                self.parameters_buffer[N:(N + N_ext)] = extension_tensor
                group["params"][0] = nn.Parameter(
                    self.parameters_buffer[:(N + N_ext)].requires_grad_(True)
                )
                self.optimizer.cpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        self.optimizer.state = self.optimizer.gpu_adam.state | self.optimizer.cpu_adam.state
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_send_to_gpui_cnt,
        new_parameters=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "parameters": new_parameters,
        }
        if self.args.offload and self.args.gpu_cache == "xyzosr":
            optimizable_tensors = self.cat_tensors_to_unified_adam(d)
        else:
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        if self.args.offload:
            if self.args.gpu_cache == "no_cache":
                self._parameters = optimizable_tensors["parameters"]
                dims = [self._xyz.shape[1], self._opacity.shape[1], self._scaling.shape[1], self._rotation.shape[1], self._features_dc.shape[1], self._features_rest.shape[1]]           
                self._xyz, self._opacity, self._scaling, self._rotation, self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1)
                
                assert self._parameters.is_pinned()
                assert self._xyz.is_pinned()
                assert self._opacity.is_pinned()
                assert self._scaling.is_pinned()
                assert self._rotation.is_pinned()
                assert self._features_dc.is_pinned()
                assert self._features_rest.is_pinned()
                
            elif self.args.gpu_cache == "xyzosr":
                self._xyz = optimizable_tensors["xyz"]
                self._opacity = optimizable_tensors["opacity"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._parameters = optimizable_tensors["parameters"]
                dims = [3, 45]           
                self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1)
                
                assert self._xyz.is_cuda
                assert self._opacity.is_cuda
                assert self._scaling.is_cuda
                assert self._rotation.is_cuda
                assert self._parameters.is_pinned()
                assert self._features_dc.is_pinned()
                assert self._features_rest.is_pinned()
            
            else:
                raise ValueError("Invalid gpu cache strategy.")
            
        else:
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.sum_visible_count_in_one_batch = torch.zeros(
            (self.get_xyz.shape[0]), device=self.device
        )

        self.send_to_gpui_cnt = torch.cat(
            (self.send_to_gpui_cnt, new_send_to_gpui_cnt), dim=0
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        # [N * number of selected points, 3]

        utils.get_log_file().write(
            "Number of split gaussians: {}\n".format(selected_pts_mask.sum().item())
        )
        
        if self.args.offload:
            selected_pts_mask_cpu = selected_pts_mask.cpu()
            if self.args.gpu_cache == "no_cache":
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
                    selected_pts_mask
                ].repeat(N, 1)
                new_scaling = self.scaling_inverse_activation(
                    self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
                )
                new_parameters = self._parameters[selected_pts_mask_cpu].repeat(N, 1)
                new_parameters[:, :self._xyz.shape[1]] = xyz
                new_parameters[:, (self._xyz.shape[1] + self._opacity.shape[1]):(self._xyz.shape[1] + self._opacity.shape[1] + self._scaling.shape[1])] = new_scaling
                new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask].repeat(N, 1)
                
                new_xyz = None
                new_features_dc = None
                new_features_rest = None
                new_opacities = None
                new_scaling = None
                new_rotation = None
                
            elif self.args.gpu_cache == "xyzosr":
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
                    selected_pts_mask
                ].repeat(N, 1)
                new_scaling = self.scaling_inverse_activation(
                    self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
                )
                new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
                new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
                new_parameters = self._parameters[selected_pts_mask_cpu].repeat(N, 1)
                new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask].repeat(N, 1)
                
                new_features_dc = None
                new_features_rest = None
            
            else:
                raise ValueError("Invalid gpu cache strategy.")
        
        else:
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
                selected_pts_mask
            ].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
            new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask].repeat(N, 1)
            new_parameters = None
            
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_send_to_gpui_cnt,
            new_parameters,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        utils.get_log_file().write(
            "Number of cloned gaussians: {}\n".format(selected_pts_mask.sum().item())
        )
        
        if self.args.offload:
            selected_pts_mask_cpu = selected_pts_mask.cpu()
            if self.args.gpu_cache == "no_cache":
                new_xyz = None
                new_features_dc = None
                new_features_rest = None
                new_opacities = None
                new_scaling = None
                new_rotation = None
                new_parameters = self._parameters[selected_pts_mask_cpu]
                new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask]
                
            elif self.args.gpu_cache == "xyzosr":
                new_xyz = self._xyz[selected_pts_mask]
                new_opacities = self._opacity[selected_pts_mask]
                new_scaling = self._scaling[selected_pts_mask]
                new_rotation = self._rotation[selected_pts_mask]
                new_parameters = self._parameters[selected_pts_mask_cpu]
                new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask]
                new_features_dc = None
                new_features_rest = None
            
            else:
                raise ValueError("Invalid gpu cache strategy.")
            
        else:
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_send_to_gpui_cnt = self.send_to_gpui_cnt[selected_pts_mask]
            new_parameters = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_send_to_gpui_cnt,
            new_parameters,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        args = utils.get_args()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # assert False, f"What's inside grad: grad.device={grad.device.type}, grad={grad.data}"
        
        # densification_stats = {}
        # densification_stats["view_space_grad"] = grads.mean().item()
        # densification_stats["view_space_grad_max"] = grads.max().item()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # NOTE: this is bug in its implementation.
            assert torch.all(
                self.max_radii2D == 0
            ), "In its implementation, max_radii2D is all 0. This is a bug."
            assert torch.all(
                big_points_vs == False
            ), "In its implementation, big_points_vs is all False. This is a bug."
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(
        self, viewspace_point_tensor, update_filter
    ):  # the :2] is a weird implementation. It is because viewspace_point_tensor is (N, 3) tensor.
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def gsplat_add_densification_stats(
        self, viewspace_point_tensor_grad, send2gpu_visibility_filter, update_filter, width, height
    ):  # the :2] is a weird implementation. It is because viewspace_point_tensor is (N, 3) tensor.
        grad = viewspace_point_tensor_grad  # (B, N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_visibility_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[send2gpu_visibility_filter] += 1

    def gsplat_add_densification_stats_exact_filter(
        self, viewspace_point_tensor_grad, send2gpu_final_filter_indices, width, height
    ):  # the :2] is a weird implementation. It is because viewspace_point_tensor is (N, 3) tensor.
        grad = viewspace_point_tensor_grad  # (N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_final_filter_indices] += torch.norm(
            grad, dim=-1, keepdim=True
        )
        self.denom[send2gpu_final_filter_indices] += 1

    def group_for_redistribution(self):
        args = utils.get_args()
        if args.gaussians_distribution:
            return utils.DEFAULT_GROUP
        else:
            return utils.SingleGPUGroup()

    def all2all_gaussian_state(self, state, destination, i2j_send_size):
        comm_group = self.group_for_redistribution()

        # state: (N, ...) tensor
        state_to_gpuj = []
        state_from_gpuj = []
        for j in range(comm_group.size()):  # ugly implementation.
            state_to_gpuj.append(state[destination == j, ...].contiguous())
            state_from_gpuj.append(
                torch.zeros(
                    (i2j_send_size[j][comm_group.rank()], *state.shape[1:]),
                    device="cuda",
                )
            )

        # print(f"before all_to_all, ws={comm_group.size()}, rank={comm_group.rank()}")

        torch.distributed.all_to_all(state_from_gpuj, state_to_gpuj, group=comm_group)

        # print(f"after all_to_all, ws={comm_group.size()}, rank={comm_group.rank()}")

        state_from_remote = torch.cat(
            state_from_gpuj, dim=0
        ).contiguous()  # it stucks at here.
        # print(f"state_from_remote, ws={comm_group.size()}, rank={comm_group.rank()}")
        return state_from_remote

    def all2all_tensors_in_optimizer_implementation_1(self, destination, i2j_send_size):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = self.all2all_gaussian_state(
                        stored_state["momentum_buffer"], destination, i2j_send_size
                    )
                else:
                    stored_state["exp_avg"] = self.all2all_gaussian_state(
                        stored_state["exp_avg"], destination, i2j_send_size
                    )
                    stored_state["exp_avg_sq"] = self.all2all_gaussian_state(
                        stored_state["exp_avg_sq"], destination, i2j_send_size
                    )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    self.all2all_gaussian_state(
                        group["params"][0], destination, i2j_send_size
                    ),
                    requires_grad=True,
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    self.all2all_gaussian_state(
                        group["params"][0], destination, i2j_send_size
                    ),
                    requires_grad=True,
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_all_optimizer_states(self):
        all_tensors = []
        all_shapes = []
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    all_tensors.append(stored_state["momentum_buffer"])
                    all_shapes.append(stored_state["momentum_buffer"].shape)
                else:
                    all_tensors.append(stored_state["exp_avg"])
                    all_shapes.append(stored_state["exp_avg"].shape)

                    all_tensors.append(stored_state["exp_avg_sq"])
                    all_shapes.append(stored_state["exp_avg_sq"].shape)

                all_tensors.append(group["params"][0])
                all_shapes.append(group["params"][0].shape)

                # release the memory BUG: release the memory will cause error. Maybe it will release memory which may use later.
                # stored_state["exp_avg"] = None
                # stored_state["exp_avg_sq"] = None
                # group["params"][0] = None

            else:
                all_tensors.append(group["params"][0])
                all_shapes.append(group["params"][0].shape)

                # release the memory BUG: release the memory will cause error. Maybe it will release memory which may use later.
                # group["params"][0] = None
        return all_tensors, all_shapes

    def update_all_optimizer_states(self, updated_tensors):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = updated_tensors.pop(
                        0
                    ).contiguous()
                else:
                    stored_state["exp_avg"] = updated_tensors.pop(0).contiguous()
                    stored_state["exp_avg_sq"] = updated_tensors.pop(0).contiguous()

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    updated_tensors.pop(0).contiguous(), requires_grad=True
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    updated_tensors.pop(0).contiguous(), requires_grad=True
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def all2all_tensors_in_optimizer_implementation_2(self, destination, i2j_send_size):
        # merge into one single all2all kernal launch.

        # get all optimizer states for all2all
        all_tensors, all_shapes = self.get_all_optimizer_states()
        # flatten all tensors with start_dim=1, then concate them at dim=1
        all_tensors_flatten = [tensor.flatten(start_dim=1) for tensor in all_tensors]
        all_tensors_catted = torch.cat(all_tensors_flatten, dim=1).contiguous()
        all_tensors_flatten = None  # release memory

        # all2all
        all_remote_tensors_catted = self.all2all_gaussian_state(
            all_tensors_catted, destination, i2j_send_size
        )
        all_tensors_catted = None  # release memory

        # split all_tensors_catted to original shapes
        all_remote_tensors_flatten = torch.split(
            all_remote_tensors_catted,
            [shape[1:].numel() for shape in all_shapes],
            dim=1,
        )
        all_remote_tensors_catted = None  # release memory
        all_remote_tensors = [
            tensor.view(tensor.shape[:1] + shape[1:])
            for tensor, shape in zip(all_remote_tensors_flatten, all_shapes)
        ]
        all_remote_tensors_flatten = None  # release memory

        # update optimizer states
        optimizable_tensors = self.update_all_optimizer_states(all_remote_tensors)
        all_remote_tensors = None

        return optimizable_tensors

    def all2all_tensors_in_optimizer(self, destination, i2j_send_size):
        return self.all2all_tensors_in_optimizer_implementation_1(
            destination, i2j_send_size
        )
        # return self.all2all_tensors_in_optimizer_implementation_2(destination, i2j_send_size)
        # when cross node all2all on perl, implementation_2 will get stuck at 1600 iterations, I do not know the reason.

    def get_destination_1(self, world_size):
        # norm p=0
        return torch.randint(0, world_size, (self.get_xyz.shape[0],), device="cuda")

    def need_redistribute_gaussians(self, group):
        args = utils.get_args()
        if group.size() == 1:
            return False
        if utils.get_denfify_iter() == args.redistribute_gaussians_frequency:
            # do redistribution after the first densification.
            return True
        local_n_3dgs = self.get_xyz.shape[0]
        all_local_n_3dgs = [None for _ in range(group.size())]
        torch.distributed.all_gather_object(all_local_n_3dgs, local_n_3dgs, group=group)
        if min(all_local_n_3dgs) * args.redistribute_gaussians_threshold < max(
            all_local_n_3dgs
        ):
            return True
        return False

    def redistribute_gaussians(self):
        args = utils.get_args()
        if args.redistribute_gaussians_mode == "no_redistribute":
            return

        comm_group_for_redistribution = self.group_for_redistribution()
        if not self.need_redistribute_gaussians(comm_group_for_redistribution):
            return

        # Get each 3dgs' destination GPU.
        if args.redistribute_gaussians_mode == "random_redistribute":
            # random redistribution to balance the number of gaussians on each GPU.
            destination = self.get_destination_1(comm_group_for_redistribution.size())
        else:
            raise ValueError(
                "Invalid redistribute_gaussians_mode: "
                + args.redistribute_gaussians_mode
            )

        # Count the number of 3dgs to be sent to each GPU.
        local2j_send_size = torch.bincount(
            destination, minlength=comm_group_for_redistribution.size()
        ).int()
        assert (
            len(local2j_send_size) == comm_group_for_redistribution.size()
        ), "local2j_send_size: " + str(local2j_send_size)

        i2j_send_size = torch.zeros(
            (
                comm_group_for_redistribution.size(),
                comm_group_for_redistribution.size(),
            ),
            dtype=torch.int,
            device="cuda",
        )
        torch.distributed.all_gather_into_tensor(
            i2j_send_size, local2j_send_size, group=comm_group_for_redistribution
        )
        i2j_send_size = i2j_send_size.cpu().numpy().tolist()
        # print("rank", utils.LOCAL_RANK, "local2j_send_size: ", local2j_send_size, "i2j_send_size: ", i2j_send_size)

        optimizable_tensors = self.all2all_tensors_in_optimizer(
            destination, i2j_send_size
        )
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.sum_visible_count_in_one_batch = torch.zeros(
            (self.get_xyz.shape[0]), device="cuda"
        )
        # NOTE: This function is called right after desify_and_prune. Therefore self.xyz_gradient_accum, self.denom and self.max_radii2D are all zero.
        # We do not need to all2all them here.

        # should I all2all send_to_gpui_cnt? I think I should not. Because 1. for simplicity now, 2. we should refresh it and do not use too old statistics.
        self.send_to_gpui_cnt = torch.zeros(
            (self.get_xyz.shape[0], comm_group_for_redistribution.size()),
            dtype=torch.int,
            device="cuda",
        )

        torch.cuda.empty_cache()


def get_sparse_ids(tensors):
    sparse_ids = None
    with torch.no_grad():
        for tensor in tensors:
            # Apply torch.nonzero()
            nonzero_indices = torch.nonzero(tensor)
            # Extract the row indices
            row_indices = nonzero_indices[:, 0]
            # Count unique rows
            if sparse_ids is None:
                sparse_ids = row_indices
            else:
                sparse_ids = torch.cat((sparse_ids, row_indices))

        sparse_ids = torch.unique(sparse_ids, sorted=True)
        return sparse_ids


def sync_gradients_sparsely(gaussians, group):
    with torch.no_grad():
        sparse_ids = get_sparse_ids(
            [gaussians._xyz.grad.data]
        )  # sparse ids are non-zero ids
        # get boolean mask of sparse ids
        sparse_ids_mask = torch.zeros(
            (gaussians._xyz.shape[0]), dtype=torch.bool, device="cuda"
        )
        sparse_ids_mask[sparse_ids] = True

        torch.distributed.all_reduce(sparse_ids_mask, op=dist.ReduceOp.SUM, group=group)

        def sync_grads(data):
            sparse_grads = data.grad.data[
                sparse_ids_mask
            ].contiguous()  # contiguous() memory is needed for collective communication.
            torch.distributed.all_reduce(
                sparse_grads, op=dist.ReduceOp.SUM, group=group
            )
            data.grad.data[sparse_ids_mask] = sparse_grads

        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)
        # We must optimize this, because there should be large kernel launch overhead.

    log_file = utils.get_log_file()
    non_zero_indices_cnt = sparse_ids_mask.sum().item()
    total_indices_cnt = sparse_ids_mask.shape[0]
    log_file.write(
        "iterations: [{}, {}) non_zero_indices_cnt: {} total_indices_cnt: {} ratio: {}\n".format(
            utils.get_cur_iter(),
            utils.get_cur_iter() + utils.get_args().bsz,
            non_zero_indices_cnt,
            total_indices_cnt,
            non_zero_indices_cnt / total_indices_cnt,
        )
    )


def sync_gradients_densely(gaussians, group):
    with torch.no_grad():

        def sync_grads(data):
            torch.distributed.all_reduce(
                data.grad.data, op=dist.ReduceOp.SUM, group=group
            )

        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)


def sync_gradients_fused_densely(gaussians, group):
    with torch.no_grad():
        # 1. cat all parameters' grad to a single tensor
        # 2. allreduce
        # 3. split the allreduced tensor to each parameter's grad
        all_params_grads = [
            param.grad.data
            for param in [
                gaussians._xyz,
                gaussians._features_dc,
                gaussians._features_rest,
                gaussians._opacity,
                gaussians._scaling,
                gaussians._rotation,
            ]
        ]
        all_params_grads_dim1 = [param_grad.shape[1] for param_grad in all_params_grads]
        catted_params_grads = torch.cat(all_params_grads, dim=1).contiguous()
        torch.distributed.all_reduce(
            catted_params_grads, op=dist.ReduceOp.SUM, group=group
        )
        split_params_grads = torch.split(
            catted_params_grads, all_params_grads_dim1, dim=1
        )
        for param_grad, split_param_grad in zip(all_params_grads, split_params_grads):
            param_grad.copy_(split_param_grad)


def sync_gradients_fused_sparsely(gaussians, group):
    raise NotImplementedError("Fused sparse sync gradients is not implemented yet.")
