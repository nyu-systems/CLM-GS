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
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
import utils.general_utils as utils
from optimizer import UnifiedAdam
import numba.cuda

from strategies.base_gaussian_model import BaseGaussianModel


class GaussianModelCLMOffload(BaseGaussianModel):
    """Gaussian model for xyzosr offload mode - xyz/opacity/scaling/rotation on GPU, features in CPU pinned memory"""

    def _get_device(self):
        return "cuda"

    def create_from_pcd_offloaded(self, pcd: BasicPointCloud, spatial_lr_scale: float, subsample_ratio=1.0):
        log_file = utils.get_log_file()
        self.spatial_lr_scale = spatial_lr_scale

        # Allocate pinned buffers for features
        parameters_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
        parameters_grad_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
        assert self.parameters_buffer.is_pinned()
        assert self.parameters_grad_buffer.is_pinned()

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to("cuda")
        fused_point_cloud = fused_point_cloud.contiguous()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to("cpu"))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to("cpu")
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
            print("Subsample ratio: ", subsample_ratio)
            print("Number of points after subsampling : ", sub_N)

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
        
        # xyz, opacity, scaling, rotation on GPU
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # features in CPU pinned memory as concatenated parameter
        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous().view(N, -1)  # (N, 1, 3) -> (N, 3)
        features_rest = features[:, :, 1:].transpose(1, 2).contiguous().view(N, -1)  # (N, 15, 3) -> (N, 45)
        dims = [features_dc.shape[1], features_rest.shape[1]]
        torch.cat((features_dc, features_rest), dim=1, out=self.parameters_buffer[:N])
        
        self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
        self._features_dc, self._features_rest = torch.split(self._parameters, dims, dim=1) 
        self.max_radii2D = torch.zeros((N), device="cuda")
        self.sum_visible_count_in_one_batch = torch.zeros((N), device="cuda")
        
        self.param_dims = torch.tensor(dims, dtype=torch.int, device='cuda')

    def all_parameters(self):
        return [
            self._xyz,
            self._opacity,
            self._scaling,
            self._rotation,
            self._parameters,
        ]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        args = utils.get_args()
        log_file = utils.get_log_file()
        
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
            fused=True if args.fused_adam == "torch_fused" else False,
            sparse=self.args.sparse_adam,
        )
                

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
                    log_file.write(
                        param_group["name"]
                        + " betas: "
                        + str(param_group["betas"])
                        + "\n"
                    )
            elif training_args.lr_scale_mode == "accumu":
                lr_scale = 1
            else:
                assert False, f"lr_scale_mode {training_args.lr_scale_mode} not supported."

        # Update columns_lr for UnifiedAdam
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

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def save_tensors(self, parent_path):
        mkdir_p(parent_path)

        torch.save(self._xyz, os.path.join(parent_path, "xyz.pt"))
        torch.save(self._opacity, os.path.join(parent_path, "opacity.pt"))
        torch.save(self._scaling, os.path.join(parent_path, "scaling.pt"))
        torch.save(self._rotation, os.path.join(parent_path, "rotation.pt"))
        torch.save(self._parameters, os.path.join(parent_path, "parameters.pt"))

    def load_tensors(self, parent_path):
        _xyz = torch.load(os.path.join(parent_path, "xyz.pt"), map_location="cpu")
        _opacity = torch.load(os.path.join(parent_path, "opacity.pt"), map_location="cpu")
        _scaling = torch.load(os.path.join(parent_path, "scaling.pt"), map_location="cpu")
        _rotation = torch.load(os.path.join(parent_path, "rotation.pt"), map_location="cpu")
        _features = torch.load(os.path.join(parent_path, "parameters.pt"), map_location="cpu")

        N = _xyz.shape[0]
        print("Number of points before initialization : ", N)

        # Handle upsampling/subsampling
        if self.args.upsample_ratio != 0.0:
            assert self.args.subsample_ratio == 1.0, "Can not upsample and subsample at the same time"

            up_N = int(N * self.args.upsample_ratio)
            print("Upsample ratio: ", self.args.upsample_ratio)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            upsampled_set_cpu, _ = torch.randperm(N)[:(up_N % N)].sort()

            _xyz_up = torch.cat([_xyz] * (up_N // N) + [_xyz[upsampled_set_cpu]])
            _opacity_up = torch.cat([_opacity] * (up_N // N) + [_opacity[upsampled_set_cpu]])
            _scaling_up = torch.cat([_scaling] * (up_N // N) + [_scaling[upsampled_set_cpu]])
            _rotation_up = torch.cat([_rotation] * (up_N // N) + [_rotation[upsampled_set_cpu]])
            _features_up = torch.cat([_features] * (up_N // N) + [_features[upsampled_set_cpu]])

            scaling_up = torch.exp(_scaling_up)
            noise = (torch.rand_like(_xyz_up) + 0.5) * torch.clamp(scaling_up, max=30)
            _xyz_up.add_(noise)
            _xyz = torch.cat((_xyz, _xyz_up))
            _opacity = torch.cat((_opacity, _opacity_up))
            _scaling = torch.cat((_scaling, _scaling_up))
            _rotation = torch.cat((_rotation, _rotation_up))
            _features = torch.cat((_features, _features_up))
            N = N + up_N
            print("Number of points after upsampling : ", _xyz.shape[0])

        elif self.args.subsample_ratio != 1.0:
            assert self.args.subsample_ratio > 0 and self.args.subsample_ratio < 1
            sub_N = int(N * self.args.subsample_ratio)
            print("Subsample ratio: ", self.args.subsample_ratio)
            print("Number of points after subsampling : ", sub_N)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            subsampled_set_cpu, _ = torch.randperm(N)[:sub_N].sort()

            _xyz = _xyz[subsampled_set_cpu]
            _opacity = _opacity[subsampled_set_cpu]
            _scaling = _scaling[subsampled_set_cpu]
            _rotation = _rotation[subsampled_set_cpu]
            _features = _features[subsampled_set_cpu]
            N = sub_N

        # Allocate pinned buffer
        parameters_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
        parameters_grad_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
        assert self.parameters_buffer.is_pinned()
        assert self.parameters_grad_buffer.is_pinned()

        # xyz, opacity, scaling, rotation on GPU
        self._xyz = nn.Parameter(_xyz.to("cuda").requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.to("cuda").requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.to("cuda").requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.to("cuda").requires_grad_(True))
        
        # features in pinned memory
        self.parameters_buffer[:N].copy_(_features)
        self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
        self._features_dc, self._features_rest = torch.split(self._parameters, [3, 45], dim=1)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def save_sub_plys(
        self, path, n_split, split_size
    ):
        args = utils.get_args()
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
        # Single GPU mode - no distribution logic needed
        _xyz = _features_dc = _features_rest = _opacity = _scaling = _rotation = None
        utils.log_cpu_memory_usage("start save_ply")
        
        # Directly use local tensors
        _xyz = self._xyz
        _features_dc = self._features_dc
        _features_rest = self._features_rest
        _opacity = self._opacity
        _scaling = self._scaling
        _rotation = self._rotation

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
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_raw_ply(path)
        N = xyz.shape[0]

        _xyz = torch.from_numpy(xyz)
        _opacity = torch.from_numpy(opacities)
        _scaling = torch.from_numpy(scales)
        _rotation = torch.from_numpy(rots)
        _features_dc = torch.from_numpy(features_dc)
        _features_rest = torch.from_numpy(features_extra)

        if self.args.upsample_ratio != 0.0:
            assert self.args.subsample_ratio == 1.0, "Can not upsample and subsample at the same time"

            up_N = int(N * self.args.upsample_ratio)
            print("Upsample ratio: ", self.args.upsample_ratio)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            upsampled_set_cpu, _ = torch.randperm(N)[:(up_N % N)].sort()

            _xyz_up = torch.cat([_xyz] * (up_N // N) + [_xyz[upsampled_set_cpu]])
            _opacity_up = torch.cat([_opacity] * (up_N // N) + [_opacity[upsampled_set_cpu]])
            _scaling_up = torch.cat([_scaling] * (up_N // N) + [_scaling[upsampled_set_cpu]])
            _rotation_up = torch.cat([_rotation] * (up_N // N) + [_rotation[upsampled_set_cpu]])
            _features_dc_up = torch.cat([_features_dc] * (up_N // N) + [_features_dc[upsampled_set_cpu]])
            _features_rest_up = torch.cat([_features_rest] * (up_N // N) + [_features_rest[upsampled_set_cpu]])

            s = torch.exp(_scaling_up) # just for noise generation. not affecting `_scaling_up`.
            noise = (torch.rand_like(_xyz_up) + 0.5) * torch.clamp(s, max=30)
            _xyz_up.add_(noise)
            _xyz = torch.cat((_xyz, _xyz_up))
            _opacity = torch.cat((_opacity, _opacity_up))
            _scaling = torch.cat((_scaling, _scaling_up))
            _rotation = torch.cat((_rotation, _rotation_up))
            _features_dc = torch.cat((_features_dc, _features_dc_up))
            _features_rest = torch.cat((_features_rest, _features_rest_up))
            N = N + up_N
            print("Number of points after upsampling : ", _xyz.shape[0])

        elif self.args.subsample_ratio != 1.0:
            assert self.args.subsample_ratio > 0 and self.args.subsample_ratio < 1
            sub_N = int(N * self.args.subsample_ratio)
            print("Subsample ratio: ", self.args.subsample_ratio)
            print("Number of points after subsampling : ", sub_N)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            subsampled_set_cpu, _ = torch.randperm(N)[:sub_N].sort()

            _xyz = _xyz[subsampled_set_cpu]
            _opacity = _opacity[subsampled_set_cpu]
            _scaling = _scaling[subsampled_set_cpu]
            _rotation = _rotation[subsampled_set_cpu]
            _features_dc = _features_dc[subsampled_set_cpu]
            _features_rest = _features_rest[subsampled_set_cpu]
            N = sub_N


        self.parameters_buffer = torch.empty(0)
        self.parameters_grad_buffer = torch.zeros(0)

        if _features_dc.ndim == 3:
            _features_dc = _features_dc.permute(0, 2, 1).reshape(N, -1)
        if _features_rest.ndim == 3:
            _features_rest = _features_rest.permute(0, 2, 1).reshape(N, -1)

        parameters_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
        parameters_grad_buffer_array = numba.cuda.pinned_array((self.args.prealloc_capacity, 48), dtype=np.float32)
        self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
        assert self.parameters_buffer.is_pinned()
        assert self.parameters_grad_buffer.is_pinned()
        # self.parameters_buffer = torch.empty((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)
        # self.parameters_grad_buffer = torch.zeros((self.args.prealloc_capacity, 48), dtype=torch.float, pin_memory=True)

        self.parameters_buffer[:N] = torch.cat((_features_dc, _features_rest), dim=1)

        self._parameters = nn.Parameter(
            self.parameters_buffer[:N].requires_grad_(True)
        )

        self._xyz = nn.Parameter(
            _xyz.to(torch.float).to(self.device).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            _opacity.to(torch.float).to(self.device).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            _scaling.to(torch.float).to(self.device).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            _rotation.to(torch.float).to(self.device).requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path):
        self.one_file_load_ply(path)

    def reset_opacity(self):
        utils.LOG_FILE.write("Resetting opacity to 0.01\n")
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_unified_adam(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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

                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))

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
        optimizable_tensors = self._prune_unified_adam(valid_points_mask)
        
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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.sum_visible_count_in_one_batch = self.sum_visible_count_in_one_batch[valid_points_mask]

    def cat_tensors_to_unified_adam(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if tensors_dict[group["name"]] is None:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]

            if group["params"][0].is_cuda:
                stored_state = self.optimizer.gpu_adam.state.get(group["params"][0], None)

                assert stored_state is not None, "Unified adam is a stateful optimizer."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (stored_state["momentum_buffer"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )

                del self.optimizer.gpu_adam.state[group["params"][0]]
                
                # Update parameters.
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            
            else:
                stored_state = self.optimizer.cpu_adam.state.get(group["params"][0], None)

                assert stored_state is not None, "Unified adam is a stateful optimizer."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (stored_state["momentum_buffer"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
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
        optimizable_tensors = self.cat_tensors_to_unified_adam(d)
        
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

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.sum_visible_count_in_one_batch = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        utils.get_log_file().write(
            "Number of split gaussians: {}\n".format(selected_pts_mask.sum().item())
        )

        selected_pts_mask_cpu = selected_pts_mask.cpu()
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_parameters = self._parameters[selected_pts_mask_cpu].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            None,
            None,
            new_opacities,
            new_scaling,
            new_rotation,
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
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        utils.get_log_file().write(
            "Number of cloned gaussians: {}\n".format(selected_pts_mask.sum().item())
        )

        selected_pts_mask_cpu = selected_pts_mask.cpu()
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_parameters = self._parameters[selected_pts_mask_cpu]

        self.densification_postfix(
            new_xyz,
            None,
            None,
            new_opacities,
            new_scaling,
            new_rotation,
            new_parameters,
        )

    def gsplat_add_densification_stats_exact_filter(
        self, viewspace_point_tensor_grad, radii, send2gpu_final_filter_indices, width, height
    ):
        self.max_radii2D[send2gpu_final_filter_indices] = torch.max(
            self.max_radii2D[send2gpu_final_filter_indices], radii
        )
        grad = viewspace_point_tensor_grad  # (N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_final_filter_indices] += torch.norm(
            grad, dim=-1, keepdim=True
        )
        self.denom[send2gpu_final_filter_indices] += 1