#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import datetime
import os

from gmpi.utils import get_config

DEFAULT_ADDR = "127.0.1.1"
DEFAULT_PORT = "8378"

CMD_GMPI = "export CUDA_LAUNCH_BLOCKING=1 && \
       export OMP_NUM_THREADS=10 && \
       export MKL_NUM_THREADS=10 && \
       export PYTHONPATH={}:$PYTHONPATH && \
       python \
       {} \
       --exp-config {} \
       --run-type {} \
       --run_dataset {} \
       --num-gpus {} \
       --master_port {} \
       --cur-time {}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--task-type",
        choices=["gmpi"],
        required=True,
        help="generative mpi's taks type",
    )
    parser.add_argument(
        "--run-type",
        choices=["train"],
        required=True,
        help="run type of the experiment (train)",
    )
    parser.add_argument(
        "--run_dataset",
        choices=["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"],
        required=True,
        help="Dataset to run.",
    )
    parser.add_argument("--master_addr", type=str, default=DEFAULT_ADDR)
    parser.add_argument("--master_port", type=str, default=DEFAULT_PORT)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

    repo_path = os.path.dirname(os.path.abspath(__file__))

    if args.task_type == "gmpi":
        print("\nIn launch, task_type: GMPI\n")
        f_script = os.path.join(repo_path, "run_gmpi.py")
        exp_config = "./configs/gmpi.yml"
    else:
        raise ValueError

    config = get_config(exp_config, None)

    if args.task_type in ["gmpi"]:
        tmp_cmd = CMD_GMPI.format(
            f"{repo_path}/gmpi/models:{repo_path}",  # the 1st is for loading pretrained wegights, it requires `torch_utils` from StyleGAN2
            f_script,
            exp_config,
            args.run_type,
            args.run_dataset,
            args.nproc_per_node,
            args.master_port,
            cur_time,
        )
    else:
        raise ValueError

    print("\n", tmp_cmd, "\n")

    os.system(tmp_cmd)
