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
import utils.general_utils as utils
from optimizer import SelectiveAdam
from strategies.base_gaussian_model import BaseGaussianModel


class GaussianModelNoOffload(BaseGaussianModel):
    """GPU-only Gaussian Model implementation"""

    def _get_device(self):
        """Return the device type for this model"""
        return "cuda"

    def __init__(self, sh_degree: int, only_for_rendering: bool = False):
        super().__init__(sh_degree, only_for_rendering)
        self.sum_visible_count_in_one_batch = torch.empty(0)

    def capture(self):
        assert False, "optimizer not fixed yet"
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
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
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, subsample_ratio=1.0):
        """Initialize from point cloud data (GPU only)"""
        log_file = utils.get_log_file()
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = (
            torch.tensor(np.asarray(pcd.points)).float()
        )
        fused_point_cloud = fused_point_cloud.contiguous()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        N = fused_point_cloud.shape[0]
        print("Number of points before initialization : ", N)

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

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

        fused_point_cloud = fused_point_cloud.to("cuda")
        features = features.to("cuda")
        scales = scales.to("cuda")

        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (N, 1), dtype=torch.float, device="cuda"
            )
        )

        args = utils.get_args()
        log_file.write("Number of initialized points: {}\n".format(N))

        if args.drop_initial_3dgs_p > 0.0:
            drop_mask = (
                np.random.rand(N) > args.drop_initial_3dgs_p
            )
            fused_point_cloud = fused_point_cloud[drop_mask]
            features = features[drop_mask]
            scales = scales[drop_mask]
            rots = rots[drop_mask]
            opacities = opacities[drop_mask]
            log_file.write(
                "Number of initialized points after random drop: {}\n".format(fused_point_cloud.shape[0])
            )

        # Init parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.sum_visible_count_in_one_batch = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def all_parameters(self):
        """Return list of all parameters"""
        return [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
        ]

    def training_setup(self, training_args):
        """Setup optimizer and training parameters (GPU only)"""
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        args = utils.get_args()
        log_file = utils.get_log_file()

        # Setup the optimizer (GPU only)
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
            self.optimizer = SelectiveAdam(l, eps=1e-15, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, fused=True)

        # Scale learning rates according to bsz
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
                assert (
                    False
                ), f"lr_scale_mode {training_args.lr_scale_mode} not supported."

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
        """Save tensors for GPU-only model"""
        mkdir_p(parent_path)

        torch.save(self._xyz, os.path.join(parent_path, "xyz.pt"))
        torch.save(self._opacity, os.path.join(parent_path, "opacity.pt"))
        torch.save(self._scaling, os.path.join(parent_path, "scaling.pt"))
        torch.save(self._rotation, os.path.join(parent_path, "rotation.pt"))
        torch.save(self._features_dc, os.path.join(parent_path, "features_dc.pt"))
        torch.save(self._features_rest, os.path.join(parent_path, "features_rest.pt"))

    def load_tensors(self, parent_path):
        """Load tensors for GPU-only model"""
        _xyz = torch.load(os.path.join(parent_path, "xyz.pt"), map_location="cpu")
        _opacity = torch.load(os.path.join(parent_path, "opacity.pt"), map_location="cpu")
        _scaling = torch.load(os.path.join(parent_path, "scaling.pt"), map_location="cpu")
        _rotation = torch.load(os.path.join(parent_path, "rotation.pt"), map_location="cpu")
        _features_dc = torch.load(os.path.join(parent_path, "features_dc.pt"), map_location="cpu")
        _features_rest = torch.load(os.path.join(parent_path, "features_rest.pt"), map_location="cpu")

        N = _xyz.shape[0]
        print("Number of points before initialization : ", N)

        self._xyz = nn.Parameter(_xyz.to("cuda").requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.to("cuda").requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.to("cuda").requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.to("cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.to("cuda").requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.to("cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def save_sub_plys(self, path, n_split, split_size):
        """Save model split into multiple PLY files (GPU-only)"""
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

            elements = np.empty(end - start, dtype=dtype_full)
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

    def reset_opacity(self):
        utils.LOG_FILE.write("Resetting opacity to 0.01\n")
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
        """Load from single PLY file"""
        path = os.path.join(folder, "point_cloud.ply")
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_raw_ply(path)
        N = xyz.shape[0]

        _xyz = torch.from_numpy(xyz)
        _opacity = torch.from_numpy(opacities)
        _scaling = torch.from_numpy(scales)
        _rotation = torch.from_numpy(rots)
        _features_dc = torch.from_numpy(features_dc)
        _features_rest = torch.from_numpy(features_extra)

        self._xyz = nn.Parameter(
            _xyz.to(torch.float).to(self.device).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            _features_dc.to(torch.float).to(self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            _features_rest.to(torch.float).to(self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
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
        """Replace tensor in optimizer"""
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
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """Prune optimizer states"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][mask]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]

                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )

                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """Prune points based on mask"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.sum_visible_count_in_one_batch = self.sum_visible_count_in_one_batch[
            valid_points_mask
        ]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate tensors to optimizer"""
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
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

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
        """Post-processing after densification"""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

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

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split Gaussians based on gradients"""
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        utils.get_log_file().write(
            "Number of split gaussians: {}\n".format(selected_pts_mask.sum().item())
        )

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

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians based on gradients"""
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

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def add_densification_stats(
        self, viewspace_point_tensor, update_filter
    ):
        """Add densification statistics"""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def gsplat_add_densification_stats(
        self, viewspace_point_tensor_grad, send2gpu_visibility_filter, update_filter, width, height
    ):
        """Add densification statistics for gsplat"""
        grad = viewspace_point_tensor_grad  # (B, N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_visibility_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[send2gpu_visibility_filter] += 1

    def packed_add_densification_stats(
        self, viewspace_point_tensor_grad, visibility, width, height
    ):
        """Add densification statistics for packed format"""
        grad = viewspace_point_tensor_grad
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[visibility] += torch.norm(
            grad[:, :2], dim=-1, keepdim=True
        )
        self.denom[visibility] += 1

    def gsplat_add_densification_stats_exact_filter(
        self, viewspace_point_tensor_grad, radii, send2gpu_final_filter_indices, width, height
    ):
        """Add densification statistics with exact filter"""
        grad = viewspace_point_tensor_grad  # (N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_final_filter_indices] += torch.norm(
            grad, dim=-1, keepdim=True
        )
        self.denom[send2gpu_final_filter_indices] += 1
