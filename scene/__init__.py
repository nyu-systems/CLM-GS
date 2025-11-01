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

import os
import random
import json
from random import randint
from torch.utils.data import Dataset
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam, predecode_dataset_to_disk, clean_up_disk, loadCam_raw_from_disk
import utils.general_utils as utils
import psutil
from scene.cameras import set_space_sort_key_dim

class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        self.train_cameras_info = None
        self.test_cameras_info = None
        log_file = utils.get_log_file()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        utils.log_cpu_memory_usage("before loading images meta data")

        if os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):  # This is the format from colmap.
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.llffhold
            )
        elif "matrixcity" in args.source_path:  # This is for matrixcity
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path,
                args.random_background,
                args.white_background,
                llffhold=args.llffhold,
            )
        else:
            raise ValueError("No valid dataset found in the source path")

        if not self.loaded_iter:
            # with open(scene_info.ply_path, "rb") as src_file, open(
            #     os.path.join(self.model_path, "input.ply"), "wb"
            # ) as dest_file:
                # dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.scene_info = scene_info # For torch dataloader, save scene_info

        # Set image size to global varaible. In case not all image sizes are identical, choose the minimum.
        orig_w, orig_h = (
            min([camera.width for camera in scene_info.train_cameras + scene_info.test_cameras]),
            min([camera.height for camera in scene_info.train_cameras + scene_info.test_cameras])
        )
        utils.set_img_size(orig_h, orig_w)
        # Dataset size in GB
        if (args.num_train_cameras > 0):
            assert args.num_test_cameras > 0, "Should set both `num_train_cameras` and `num_test_cameras`"
            assert args.num_train_cameras <= len(scene_info.train_cameras) and args.num_test_cameras <= len(scene_info.test_cameras), "Can not config more cameras than dataset size"
            dataset_size_in_GB = (
                1.0
                * (args.num_train_cameras + args.num_test_cameras)
                * orig_w
                * orig_h
                * 3
                / 1e9
            )
        else:
            dataset_size_in_GB = (
                1.0
                * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
                * orig_w
                * orig_h
                * 3
                / 1e9
            )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")
        
        # Preprocess dataset
        # Train on original resolution, no downsampling in our implementation.
        assert not (args.decode_dataset_to_disk and args.preload_dataset_to_gpu_threshold > 0), "Can not preload dataset to gpu and offload it to disk at the same time. Disable `decode_dataset_to_disk` or reset `preload_dataset_to_gpu_threshold`."
        if (
            dataset_size_in_GB < args.preload_dataset_to_gpu_threshold
        ):  # 10GB memory limit for dataset
            log_file.write(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable distributed_dataset_storage.\n"
            )
            print(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable distributed_dataset_storage."
            )
            args.preload_dataset_to_gpu = True
            args.distributed_dataset_storage = False

        if args.decode_dataset_to_disk:
            # Predecode dataset as raw files to local disk
            statvfs = os.statvfs(args.decode_dataset_path)
            self.decode_dataset_path = os.path.join(args.decode_dataset_path, "dataset_raw")
            
            if (not args.reuse_decoded_dataset) or (not os.path.isdir(os.path.join(args.decode_dataset_path, 'dataset_raw'))):
                available_space_in_GB = 1.0 * statvfs.f_frsize * statvfs.f_bavail / 1e9
                assert available_space_in_GB >= dataset_size_in_GB, f"Not enough space in disk for decompressed dataset. avail: {available_space_in_GB}. need: {dataset_size_in_GB}"
                log_file.write(f"[NOTE]: Pre-decoding dataset({dataset_size_in_GB}GB) to disk dir: {self.decode_dataset_path}\n")
                os.makedirs(os.path.join(self.decode_dataset_path, "dataset_raw"), exist_ok=True)
                do_decode = True
            else:
                log_file.write(f"[NOTE]: Reusing decoded dataset({dataset_size_in_GB}GB) in disk dir: {self.decode_dataset_path}\n")
                utils.print_rank_0("Reusing decoded datase on disk.")
                do_decode = False

            self.train_cameras = None
            self.test_cameras = None
            if args.num_train_cameras >= 0:
                train_cameras = scene_info.train_cameras[: args.num_train_cameras]
            else:
                train_cameras = scene_info.train_cameras
            if do_decode:
                utils.print_rank_0("Decoding Training Cameras To Disk")
                predecode_dataset_to_disk(train_cameras, args)
            self.train_cameras_info = train_cameras
            
            if len(train_cameras) > 0:
                log_file.write(
                    "Train Image size: {}x{}\n".format(
                        orig_h,
                        orig_w
                    )
                )
            
            if args.eval:
                if args.num_test_cameras >= 0:
                    test_cameras = scene_info.test_cameras[: args.num_test_cameras]
                else:
                    test_cameras = scene_info.test_cameras
                if do_decode:
                    utils.print_rank_0("Decoding Test Cameras To Disk")
                    predecode_dataset_to_disk(test_cameras, args)
                self.test_cameras_info = test_cameras
                
                if len(test_cameras) > 0:
                    log_file.write(
                        "Test Image size: {}x{}\n".format(
                            orig_h,
                            orig_w
                        )
                    )
        
        else:
            # Decode dataset in memory
            utils.print_rank_0("Decoding Training Cameras")
            self.train_cameras = None
            self.test_cameras = None
            if args.num_train_cameras >= 0:
                train_cameras = scene_info.train_cameras[: args.num_train_cameras]
            else:
                train_cameras = scene_info.train_cameras
            self.train_cameras = cameraList_from_camInfos(train_cameras, args)
            self.train_cameras_info = train_cameras
            # output the number of cameras in the training set and image size to the log file
            log_file.write(
                "Number of local training cameras: {}\n".format(len(self.train_cameras))
            )
            if len(self.train_cameras) > 0:
                log_file.write(
                    "Image size: {}x{}\n".format(
                        orig_h,
                        orig_w
                    )
                )

            if args.eval:
                utils.print_rank_0("Decoding Test Cameras")
                if args.num_test_cameras >= 0:
                    test_cameras = scene_info.test_cameras[: args.num_test_cameras]
                else:
                    test_cameras = scene_info.test_cameras
                self.test_cameras = cameraList_from_camInfos(test_cameras, args)
                self.test_cameras_info = test_cameras
                
                # output the number of cameras in the training set and image size to the log file
                log_file.write(
                    "Number of local test cameras: {}\n".format(len(self.test_cameras))
                )
                if len(self.test_cameras) > 0:
                    log_file.write(
                        "Image size: {}x{}\n".format(
                            orig_h,
                            orig_w
                        )
                    )

        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif args.load_pt_path != '':
            self.gaussians.load_tensors(args.load_pt_path)
        elif args.load_ply_path != '':
            self.gaussians.load_ply(args.load_ply_path)
        else:
            if args.offload:
                self.gaussians.create_from_pcd_offloaded(scene_info.point_cloud, self.cameras_extent, args.subsample_ratio)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.subsample_ratio)

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

        # get the longest axis in self.gaussians
        longest_axis = (self.gaussians._xyz.max(0)[0] - self.gaussians._xyz.min(0)[0]).argmax().item()
        # import pdb; pdb.set_trace()
        set_space_sort_key_dim(longest_axis)

    def save_tensors(self, iteration):
        parent_path = os.path.join(
            self.model_path, f"saved_tensors/iteration_{iteration}"
        )
        self.gaussians.save_tensors(parent_path)
    
    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        avail_ram_bytes = psutil.virtual_memory().available
        N = self.gaussians._xyz.shape[0]
        required_bytes = 16 * N * 59 * 4

        # Check if the available ram can fit all attributes for cat and map with 20% redundency.
        if avail_ram_bytes * 0.8 > required_bytes:
            # Save in one ply file.
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            # Save in multiple sub files.
            n_split = int((required_bytes + avail_ram_bytes * 0.8 - 1) // (avail_ram_bytes * 0.8))
            split_size = (N + n_split - 1) // n_split
            utils.print_rank_0(f"Requires {required_bytes / 1024 / 1024 / 1024:.3f} GB RAM for saving. Avail {avail_ram_bytes / 1024 / 1024 / 1024:.3f} GB. Split: {N} -> {n_split} x {split_size}")
            self.gaussians.save_sub_plys(os.path.join(point_cloud_path, "point_cloud.ply"), n_split, split_size)   

    def getTrainCameras(self):
        return self.train_cameras

    def getTrainCamerasInfo(self):
        return self.train_cameras_info

    def getTestCameras(self):
        return self.test_cameras
    
    def getTestCamerasInfo(self):
        return self.test_cameras_info

    def log_scene_info_to_file(self, log_file, prefix_str=""):

        # Print shape of gaussians parameters.
        log_file.write("xyz shape: {}\n".format(self.gaussians._xyz.shape))
        log_file.write("f_dc shape: {}\n".format(self.gaussians._features_dc.shape))
        log_file.write("f_rest shape: {}\n".format(self.gaussians._features_rest.shape))
        log_file.write("opacity shape: {}\n".format(self.gaussians._opacity.shape))
        log_file.write("scaling shape: {}\n".format(self.gaussians._scaling.shape))
        log_file.write("rotation shape: {}\n".format(self.gaussians._rotation.shape))

    def clean_up(self):
        # Remove the predecoded dataset from disk
        if self.args.decode_dataset_to_disk and not self.args.reuse_decoded_dataset:
            clean_up_disk(self.args)
            utils.print_rank_0("Cleaned up decoded dataset on disk.")

class SceneDataset:
    def __init__(self, cameras, cameras_info=None):
        self.cameras = cameras
        self.cameras_info = cameras_info
        self.camera_size = len(self.cameras) if self.cameras is not None else len(self.cameras_info)

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid):
        args = utils.get_args()
        if len(self.cur_epoch_cameras) == 0:
            # start a new epoch
            self.cur_epoch_cameras = list(range(self.camera_size))
            random.shuffle(self.cur_epoch_cameras)

        self.cur_iteration += 1

        if args.decode_dataset_to_disk:
            idx = 0
            while self.cur_epoch_cameras[idx] in batched_cameras_uid:
                idx += 1
            camera_idx = self.cur_epoch_cameras.pop(idx)
            viewpoint_cam = loadCam_raw_from_disk(args, camera_idx, self.cameras_info[camera_idx], to_gpu=True)
        else:
            idx = 0
            while self.cameras[self.cur_epoch_cameras[idx]].uid in batched_cameras_uid:
                idx += 1
            camera_idx = self.cur_epoch_cameras.pop(idx)
            viewpoint_cam = self.cameras[camera_idx]
        return camera_idx, viewpoint_cam

    def get_batched_cameras(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        for i in range(batch_size):
            _, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras.append(camera)
            batched_cameras_uid.append(camera.uid)

        return batched_cameras

    def get_batched_cameras_idx(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras_idx = []
        batched_cameras_uid = []
        for i in range(batch_size):
            idx, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(camera.uid)
            batched_cameras_idx.append(idx)

        return batched_cameras_idx

    def get_batched_cameras_from_idx(self, idx_list):
        return [self.cameras[i] for i in idx_list]

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []

def custom_collate_fn(batch):
    return batch

class TorchSceneDataset(Dataset):
    def __init__(self, cameras, cameras_info):
        self.cameras = cameras
        self.cameras_info = cameras_info
        self.camera_size = len(self.cameras) if self.cameras is not None else len(self.cameras_info)

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []
    
    def __len__(self):
        return self.camera_size

    def __getitem__(self, id):
        if (self.args.decode_dataset_to_disk):
            return loadCam_raw_from_disk(
                self.args,
                id,
                self.cameras_info[id],
            )
        else:
            return loadCam(
                self.args,
                id, #TODO: arg `id` in `loadCam()` is supposed to be the index inside a batch, not the global index in dataset. Passing `id` is meaningless.
                self.cameras_info[id],
                decompressed_image=None,
                return_image=False,
            )

    @property 
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_batched_cameras(self, batch_size):
        assert False, "Not implemented with torch DataLoader."

    def get_batched_cameras_idx(self, batch_size):
        assert False, "Not implemented with torch DataLoader."

    def get_batched_cameras_from_idx(self, idx_list):
        assert False, "Not implemented with torch DataLoader."

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []