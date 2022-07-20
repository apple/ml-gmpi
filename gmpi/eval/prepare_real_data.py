import argparse
import copy
import os
import random

import numpy as np
import torch
import tqdm
from torchvision.utils import save_image

import gmpi.curriculums as curriculums
import gmpi.datasets as datasets
from gmpi.train_helpers import modify_curriculums
from gmpi.utils import convert_cfg_to_dict, get_config


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in tqdm.tqdm(range(num_imgs // batch_size)):
        real_imgs, _, _, _, _ = next(dataloader)

        for img in real_imgs:
            save_image(
                img,
                os.path.join(real_dir, f"{img_counter:0>5}.png"),
                normalize=True,
                value_range=(-1, 1),
            )
            img_counter += 1


def setup_evaluation(task_name, dataset_name, save_dir, target_size=128, num_imgs=8000, **kwargs):
    if dataset_name in ["MetFaces"]:
        num_imgs = min(2048, num_imgs)

    # Only make real images if they haven't been made yet
    folder_name = f"{dataset_name}_real_res_{target_size}_n_{num_imgs}"
    real_dir = os.path.join(save_dir, task_name, folder_name)
    print("\nreal_dir: ", real_dir, "\n")

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs["img_size"] = target_size
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, **new_kwargs)
        print("outputting real images...")
        output_real_images(dataloader, num_imgs, real_dir)
        print("...done")

    return real_dir


def main(exp_config_f, dataset_name, save_dir, opts=None, n_imgs=50000):

    assert dataset_name in ["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"], f"{dataset_name}"

    config = get_config(exp_config_f, opts)

    config.defrost()
    config.GMPI.TRAIN.dataset = dataset_name
    config.freeze()

    modify_curriculums(config, flag_eval=True)

    curriculum = getattr(curriculums, config.GMPI.TRAIN.curriculum)

    metadata = curriculums.extract_metadata(curriculum, 0)

    # NOTE: for condition on pose
    metadata["sphere_center"] = config.GMPI.MPI.CAM_SETUP.cam_sphere_center_z
    metadata["sphere_r"] = config.GMPI.MPI.CAM_SETUP.cam_sphere_r
    metadata["flat_pose_dim"] = config.GMPI.TRAIN.D_cond_pose_dim

    metadata["batch_size"] = 4

    print("\nconfig: ", config, "\n")
    print("\nmetadata: ", metadata, "\n")
    print("\n", metadata["dataset_path"], "\n")

    real_dir = setup_evaluation(
        dataset_name,
        metadata["dataset"],
        save_dir,
        target_size=metadata["eval_img_size"],
        num_imgs=n_imgs,
        **metadata,
    )

    torch.save(
        {"config": convert_cfg_to_dict(config)},
        os.path.join(os.path.dirname(real_dir), "config.pth"),
    )

    with open(os.path.join(os.path.dirname(real_dir), "options.txt"), "w") as f:
        f.write(str(curriculum))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_dir", type=str, default="imgs")
    # parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--n_imgs", type=int, default=0)
    parser.add_argument("--exp_config", type=str, default="./configs/pi_gan_with_mpi.yml")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args.exp_config, args.dataset, args.save_dir, opts=None, n_imgs=args.n_imgs)
