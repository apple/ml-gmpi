#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from gmpi.utils import get_config, update_config_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--run_dataset",
        choices=["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"],
        required=True,
        help="Dataset to run.",
    )
    parser.add_argument("--cur-time", type=str, required=True, help="timestamp for current executing.")
    parser.add_argument("--num-gpus", type=int, required=True, help="#GPUs.")
    parser.add_argument("--master_port", type=str, required=True, help="#GPUs.")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    run_exp(**vars(args))


def run_exp(
    exp_config: str,
    run_type: str,
    num_gpus: int,
    cur_time: str,
    master_port: str,
    run_dataset: str,
    opts=None,
) -> None:  # cur_time: str,
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)

    log_folder_name = "seed_{}-dist_{}-{}-torgba_cond_{}-embed_func_{}-{}".format(
        config.SEED,
        int(num_gpus),
        run_dataset,
        config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc,
        config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func,
        cur_time,
    )
    log_dir = os.path.join(config.LOG_DIR, log_folder_name)

    config = update_config_log(config, run_type, log_dir)

    # add repo root
    repo_path = os.path.dirname(os.path.abspath(__file__))
    config.defrost()

    # config.REPO_ROOT = repo_path

    # sync values in config
    config.GMPI.TRAIN.port = str(config.GMPI.DDP_TRAIN.port)
    config.GMPI.TRAIN.output_dir = config.LOG_DIR

    config.freeze()

    # reproducibility set up
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    from gmpi.train import train

    mp.spawn(train, args=(num_gpus, config, master_port, run_dataset), nprocs=num_gpus, join=True)


if __name__ == "__main__":

    main()
