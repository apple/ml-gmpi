"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import argparse
import copy
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import gmpi.datasets as datasets
import gmpi.utils.pytorch_fid.fid_score as fid_score

# from pytorch_fid import fid_score


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in tqdm(range(num_imgs // batch_size)):
        real_imgs, _, _, _, _ = next(dataloader)

        for img in real_imgs:
            save_image(
                img,
                os.path.join(real_dir, f"{img_counter:0>5}.png"),
                normalize=True,
                range=(-1, 1),
            )
            img_counter += 1


def setup_evaluation(dataset_name, generated_dir, target_size=128, num_imgs=8000, debug=False, **kwargs):
    if debug:
        num_imgs = 256
    else:
        if dataset_name in ["MetFaces"]:
            num_imgs = 2048
    # Only make real images if they haven't been made yet
    real_dir = os.path.join("EvalImages", f"{dataset_name}_real_images_{str(target_size)}")
    print("\nreal_dir: ", real_dir, "\n")
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs["img_size"] = target_size
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, **new_kwargs)
        print("outputting real images...")
        output_real_images(dataloader, num_imgs, real_dir)
        print("...done")

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def output_images(
    generator,
    mpi_renderer,
    input_metadata,
    rank,
    world_size,
    output_dir,
    debug=False,
    num_imgs=2048,
    xyz_ret_single_res=True,
    use_normalized_xyz=True,
    truncation_psi=1.0,
):
    if debug:
        num_imgs = 256

    metadata = copy.deepcopy(input_metadata)
    metadata["img_size"] = metadata["eval_img_size"]
    metadata["batch_size"] = 1

    metadata["h_stddev"] = metadata.get("h_stddev_eval", metadata["h_stddev"])
    metadata["v_stddev"] = metadata.get("v_stddev_eval", metadata["v_stddev"])
    # metadata["sample_dist"] = metadata.get("sample_dist_eval", metadata["sample_dist"])
    # metadata["psi"] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    mpi_tex_pix_xyz, mpi_tex_pix_normalized_xyz = mpi_renderer.get_xyz(
        metadata["tex_size"], metadata["tex_size"], ret_single_res=xyz_ret_single_res
    )
    if use_normalized_xyz:
        stylegan2_mpi_xyz_input = mpi_tex_pix_normalized_xyz
    else:
        stylegan2_mpi_xyz_input = mpi_tex_pix_xyz

    if rank == 0:
        pbar = tqdm("generating images", total=num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn(
                (metadata["batch_size"], generator.module.z_dim),
                device=generator.module.device,
            )

            generated_imgs = []
            batch_mpi_rgbas = generator.module.forward(
                z=z,
                c=None,
                mpi_xyz_coords=stylegan2_mpi_xyz_input,
                xyz_coords_only_z=False,
                n_planes=stylegan2_mpi_xyz_input[4].shape[0],
                truncation_psi=truncation_psi,
            )
            generated_imgs, _, _, _ = mpi_renderer.render(
                batch_mpi_rgbas,
                metadata["img_size"],
                metadata["img_size"],
                horizontal_mean=metadata["h_mean"],
                horizontal_std=metadata["h_stddev"],
                vertical_mean=metadata["v_mean"],
                vertical_std=metadata["v_stddev"],
            )

            for img in generated_imgs:
                save_image(
                    img,
                    os.path.join(output_dir, f"{img_counter:0>5}.png"),
                    normalize=True,
                    range=(-1, 1),
                )
                img_counter += world_size
                if rank == 0:
                    pbar.update(world_size)
    if rank == 0:
        pbar.close()


def calculate_fid(dataset_name, generated_dir, target_size=256):
    real_dir = os.path.join("EvalImages", f"{dataset_name}_real_images_{str(target_size)}")
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, "cuda", 2048)
    torch.cuda.empty_cache()

    return fid, real_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CelebA")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_imgs", type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)
