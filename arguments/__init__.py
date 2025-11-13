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

from argparse import ArgumentParser, Namespace
import sys
import os
import utils.general_utils as utils


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    type_to_use = int
                    if len(value) > 0:
                        type_to_use = type(value[0])
                    group.add_argument(
                        "--" + key, default=value, nargs="+", type=type_to_use
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class AuxiliaryParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # ====================================================================
        # OFFLOADING CONFIGURATION
        # ====================================================================
        # There are two main offloading modes:
        # 1. BRAINDEATH OFFLOAD: Simple baseline that loads ALL parameters to GPU,
        #    processes all cameras, then offloads ALL gradients back to CPU
        # 2. FINAL OFFLOAD (PIPELINED): Advanced retention-based offload with
        #    selective loading and overlapped communication/computation
        # ====================================================================
        
        # --- Braindeath Offload (Simple Baseline) ---
        self.naive_offload = False  # Enable braindeath offload mode
        
        # --- Final Offload (Pipelined/Retention-based) ---
        self.clm_offload = False             # Enable final offload mode (required for pipelined_offload)
        self.comm_stream_priority = -1 # by default, use -1 as the priority of the stream
        self.grid_size_H = 32            # Grid size for height dimension (used in filtering/spatial hashing)
        self.grid_size_D = 32            # Grid size for depth dimension
        self.prealloc_capacity = 5_000_000  # Pre-allocated capacity for parameters

        # --- No Offload (GPU Baseline) ---
        self.no_offload = False          # Enable no offload mode

        # --- Shared Offload Optimization Flags (used by both modes) ---
        self.sparse_adam = False         # Use sparse Adam optimizer (only update visible Gaussians)
        self.fused_adam = "default"      # Fused Adam implementation: "default" or other variants
        self.delay_cpuadam_notaccessed_gs = False  # Delay CPU Adam updates for non-accessed Gaussians
        self.reorder_by_min_sparsity_at_end = True # Reorder Gaussians by minimum sparsity at end of training
        
        # ====================================================================
        # MEMORY MANAGEMENT
        # ====================================================================
        
        # ====================================================================
        # DATA LOADING & DATASET
        # ====================================================================
        self.decode_dataset_path = ""  # Path for decoded dataset storage
        self.num_workers = 0             # Number of worker threads for data loading
        self.sharing_strategy = "default" # PyTorch multiprocessing sharing strategy: "default" ("file_descriptor"), or "file_system"
        self.llffhold = 8                # LLFF dataset hold-out value
        
        # ====================================================================
        # MODEL I/O
        # ====================================================================
        self.load_ply_path = ""          # Path to load pre-trained PLY file
        self.load_ply_max = 1_000_000    # Maximum number of points to load from PLY
        self.load_pt_path = ""           # Path to load PyTorch checkpoint
        self.dense_ply_file = ""         # Path to dense PLY file
        self.start_checkpoint = ""       # Checkpoint to resume training from
        self.auto_start_checkpoint = False  # Automatically find and load latest checkpoint
        
        # ====================================================================
        # LOGGING & MONITORING
        # ====================================================================
        self.log_folder = "/tmp/gaussian_splatting"  # Folder for logging outputs
        self.log_interval = 250          # Iteration interval for logging
        self.quiet = False               # Suppress verbose output
        
        # ====================================================================
        # TRAINING CONTROL & EVALUATION
        # ====================================================================
        self.test_iterations = [7_000, 30_000]     # Iterations at which to run test evaluation
        self.save_iterations = []     # Iterations at which to save model
        self.checkpoint_iterations = []  # Iterations at which to save checkpoints
        
        # ====================================================================
        # OPTIMIZATION & LOSS
        # ====================================================================
        self.fused_loss = "advanced_fuse" # Loss fusion mode: "default", "hand_written", "torch_compile", "advanced_fuse"
        
        # ====================================================================
        # DEBUGGING & PROFILING
        # ====================================================================
        self.debug_from = -1             # Start debugging from this iteration
        self.detect_anomaly = False      # Enable PyTorch anomaly detection
        
        # ====================================================================
        # HARDWARE & DEVICE
        # ====================================================================
        self.gpu = 0                     # GPU device ID to use
        
        # ====================================================================
        # DATASET-SPECIFIC FLAGS
        # ====================================================================
        self.matrixcity_ocean_mask = False  # Enable ocean masking for MatrixCity dataset
        
        # ====================================================================
        # EXPERIMENTAL / ADVANCED FLAGS
        # ====================================================================
        self.packed = False              # Use packed representation

        self.num_save_images_during_eval = 0
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args): 
        g = super().extract(args)
        return g


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.radius_clip = 0.0
        self._source_path = ""
        self._model_path = "/tmp/gaussian_splatting"
        self._images = "images"
        self._white_background = False
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.lr_scale_loss = 1.0
        self.lr_scale_pos_and_scale = 1.0
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_memory_limit_percentage = 0.9
        self.disable_auto_densification = False
        self.random_background = False
        self.min_opacity = 0.005
        self.lr_scale_mode = "sqrt"  # can be "linear", "sqrt", or "accumu"

        # Dataset and Model save
        self.bsz = 1  # batch size.
        self.multiprocesses_image_loading = False # Disable multiprocess image loading by default to avoid out of shared memory
        self.num_train_cameras = -1
        self.num_test_cameras = -1

        self.adam_type = "default_adam"  # "default_adam", "cpu_adam"

        self.exact_filter = True
        self.log_cpu_adam_trailing_overhead = False
        self.cpu_adam_trailing_overhead = {"step": 1, "from_default_stream": 0.0, "from_comm_stream": 0.0}

        self.max_num_images_to_evaluate = int(1e9) # maximum number of images to evaluate during testing. 
        super().__init__(parser, "Optimization Parameters")

class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.enable_timer = False  # Log running time from python side.
        self.end2end_time = True  # Log end2end training time.
        self.check_gpu_memory = False  # check gpu memory usage.
        self.check_cpu_memory = False  # check cpu memory usage.
        self.log_memory_summary = False
        self.trace_cuda_mem = False

        super().__init__(parser, "Benchmark Parameters")


class DebugParams(ParamGroup):
    def __init__(self, parser):
        self.stop_update_param = (
            False  # stop updating parameters. No optimizer.step() will be called.
        )
        self.time_image_loading = False  # Log image loading time.

        self.nsys_profile = False  # profile with nsys.
        self.nsys_profile_start_iter = 1  # profile with nsys start iteration.
        self.nsys_profile_end_iter = 1000000 # profile with nsys end iteration.
        self.drop_initial_3dgs_p = 0.0  # profile with nsys.
        self.drop_duplicate_gaussians_coeff = 1.0
        self.do_not_save = False # Do not save model
        self.reset_each_iter = False # Reset max memory for  each iteration
        self.save_tensors = False # Save model parameters as .pt file

        self.reinit_ply = False

        super().__init__(parser, "Debug Parameters")


def get_combined_args(parser: ArgumentParser, auto_find_cfg_args_path=False):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        if auto_find_cfg_args_path:
            if hasattr(args_cmdline, "load_ply_path"):
                path = args_cmdline.load_ply_path
                while not os.path.exists(
                    os.path.join(path, "cfg_args")
                ) and os.path.exists(path):
                    path = os.path.join(path, "..")
                cfgfilepath = os.path.join(path, "cfg_args")
        else:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def print_all_args(args, log_file):
    # print all arguments in a readable format, each argument in a line.
    log_file.write("arguments:\n")
    log_file.write("-" * 30 + "\n")
    for arg in vars(args):
        log_file.write("{}: {}\n".format(arg, getattr(args, arg)))
    log_file.write("-" * 30 + "\n\n")
    log_file.write("bsz: " + str(args.bsz) + "\n")


def find_latest_checkpoint(log_folder):
    checkpoint_folder = os.path.join(log_folder, "checkpoints")
    if os.path.exists(checkpoint_folder):
        all_sub_folders = os.listdir(checkpoint_folder)
        if len(all_sub_folders) > 0:
            all_sub_folders.sort(key=lambda x: int(x), reverse=True)
            return os.path.join(checkpoint_folder, all_sub_folders[0])
    return ""


def init_args(args):

    assert sum([args.clm_offload, args.naive_offload, args.no_offload]) == 1, "Exactly one of clm_offload, naive_offload, or no_offload must be True"

    # Logging are saved with where model is saved.
    args.log_folder = args.model_path

    if args.auto_start_checkpoint:
        args.start_checkpoint = find_latest_checkpoint(args.log_folder)

    # sort test_iterations
    args.test_iterations.sort()
    args.save_iterations.sort()
    if len(args.save_iterations) > 0 and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.sort()

    # Set up global args
    utils.set_args(args)
