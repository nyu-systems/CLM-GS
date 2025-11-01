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
import torch
import sys
import json
from utils.general_utils import safe_state
import utils.general_utils as utils
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import train_internal
import psutil

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    args = utils.get_args()

    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    with open(args.log_folder + "/args.json", "w") as f:
        json.dump(vars(args), f)

    # create cuda trace dump folder
    if args.trace_cuda_mem:
        os.makedirs(os.path.join(args.model_path, "trace_dump"))
    

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(
        args.log_folder + "/python.log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    p = psutil.Process()
    log_file.write(f"Initial pinned memory: {p.memory_info().shared / 1024 / 1024 / 1024} GB\n")

    train_internal.training(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )

    # All done
    utils.print_rank_0("\nTraining complete.")
