import argparse
import os
import random

import numpy as np
import torch
import tqdm
from PIL import Image

import gmpi.curriculums as curriculums
from gmpi.core.mpi_renderer import MPIRenderer
from gmpi.train_helpers import modify_curriculums
from gmpi.utils import Config, convert_cfg_to_dict, get_config


def generate_img(
    metadata,
    generator,
    mpi_renderer,
    z,
    n_imgs,
    mpi_xyz_input,
    mpi_xyz_only_z=False,
    mpi_z_interpolation_ws=None,
    n_planes=32,
    disable_tqdm=False,
    light_render=None,
    stylegan2_sanity_check=False,
    verbose=False,
    truncation_psi=1.0,
    **kwargs,
):

    with torch.no_grad():

        img_list = []
        depth_map_list = []
        cam_angle_list = []

        # [B, #planes, 4, H, W]
        batch_mpi_rgbas = generator(
            z,
            None,
            mpi_xyz_input,
            mpi_xyz_only_z,
            n_planes,
            z_interpolation_ws=mpi_z_interpolation_ws,
            truncation_psi=truncation_psi,
        )

        if stylegan2_sanity_check:
            # we make all alphas full while
            mpi_rgb = batch_mpi_rgbas[:, :, :3, ...]
            old_mpi_alpha = batch_mpi_rgbas[:, :, 3:, ...]
            mpi_alpha = torch.ones_like(old_mpi_alpha)
            batch_mpi_rgbas = torch.cat((mpi_rgb, mpi_alpha), dim=2)

        if n_imgs > 1:
            # [B, #planes, 4, H, W]
            # print("\nbatch_mpi_rgbas: ", batch_mpi_rgbas.shape, "\n")
            tmp_bs, tmp_n_planes, _, tmp_h, tmp_w = batch_mpi_rgbas.shape
            batch_mpi_rgbas = batch_mpi_rgbas.unsqueeze(1).expand(-1, n_imgs, -1, -1, -1, -1)
            batch_mpi_rgbas = batch_mpi_rgbas.reshape((tmp_bs * n_imgs, tmp_n_planes, 4, tmp_h, tmp_w))

        # angles: [pitch, yaw], [N, 2]
        img, depth_map, g_c2w_mats, g_angles = mpi_renderer.render(
            batch_mpi_rgbas,
            metadata["img_size"],
            metadata["img_size"],
        )

        img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
        img = np.clip((img + 1) / 2.0, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

        # img = img.detach().cpu()

        depth_map = depth_map.permute(0, 2, 3, 1).detach().cpu().numpy()
        img_list.append(img)
        depth_map_list.append(depth_map)
        cam_angle_list.append(g_angles.cpu().numpy())

        if verbose:
            print("\nlog: ", batch_mpi_rgbas.shape, img.shape, depth_map.shape, "\n")

    return img_list, depth_map_list, cam_angle_list


def main(opt):

    os.makedirs(opt.save_dir, exist_ok=True)

    if os.path.splitext(opt.exp_config)[1] in [".yml", ".yaml"]:
        config = get_config(opt.exp_config, None)
    elif os.path.splitext(opt.exp_config)[1] == ".pth":
        config = Config(init_dict=torch.load(opt.exp_config, map_location="cpu")["config"])
    else:
        raise ValueError

    modify_curriculums(config, flag_eval=True)

    curriculum = getattr(curriculums, config.GMPI.TRAIN.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    if opt.task == "geometry":
        # NOTE: Deep3DFaceRecon can only provide mask and depth of 224x224
        # https://github.com/sicxu/Deep3DFaceRecon_pytorch
        metadata["img_size"] = 224
    
    if config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc == "none":
        # Vanilla version. The number of planes must be same in training and evaluation.
        n_mpi_planes = config.GMPI.MPI.n_gen_planes
    else:
        n_mpi_planes = opt.nplanes

    mpi_renderer = MPIRenderer(
        n_mpi_planes=n_mpi_planes,  # config.GMPI.MPI.n_gen_planes,
        plane_min_d=metadata["ray_start"],
        plane_max_d=metadata["ray_end"],
        plan_spatial_enlarge_factor=config.GMPI.MPI.CAM_SETUP.spatial_enlarge_factor,
        plane_distances_sample_method=config.GMPI.MPI.distance_sample_method,
        cam_fov=metadata["fov"],
        sphere_center_z=config.GMPI.MPI.CAM_SETUP.cam_sphere_center_z,
        sphere_r=config.GMPI.MPI.CAM_SETUP.cam_sphere_r,
        horizontal_mean=metadata["h_mean"],
        horizontal_std=metadata["h_stddev"],
        vertical_mean=metadata["v_mean"],
        vertical_std=metadata["v_stddev"],
        cam_pose_n_truncated_stds=config.GMPI.MPI.CAM_SETUP.cam_pose_n_truncated_stds,
        cam_sample_method=config.GMPI.MPI.CAM_SETUP.cam_pose_sample_method,
        mpi_align_corners=config.GMPI.MPI.align_corners,
        use_xyz_ztype=config.GMPI.TRAIN.use_xyz_ztype,
        use_normalized_xyz=config.GMPI.TRAIN.use_normalized_xyz,
        normalized_xyz_range=config.GMPI.TRAIN.normalized_xyz_range,
        use_confined_volume=config.GMPI.MPI.use_confined_volume,
        device=device,
    )

    n_src_planes = config.GMPI.MPI.n_gen_planes
    n_tgt_planes = opt.nplanes
    mpi_z_interpolation_ws = mpi_renderer.get_xyz_interpolate_ws(n_src_planes, n_tgt_planes).to(device)

    mpi_return_single_res_xyz = False
    mpi_xyz_only_z = False

    mpi_renderer.set_cam(metadata["fov"], metadata["img_size"], metadata["img_size"])
    # [#planes, tex_h, tex_w, 4]
    mpi_tex_pix_xyz, mpi_tex_pix_normalized_xyz = mpi_renderer.get_xyz(
        metadata["tex_size"],
        metadata["tex_size"],
        ret_single_res=mpi_return_single_res_xyz,
        only_z=mpi_xyz_only_z,
    )

    if config.GMPI.TRAIN.use_normalized_xyz:
        mpi_xyz_input = mpi_tex_pix_normalized_xyz
    else:
        mpi_xyz_input = mpi_tex_pix_xyz

    if mpi_xyz_only_z:
        print("\nmpi_xyz_input: ", mpi_xyz_input[4][:, 0, 0, 0], "\n")
    else:
        print("\nmpi_xyz_input: ", mpi_xyz_input[4][:, 0, 0, 2], "\n")

    print("\nconfig: ", config, "\n")
    print("\nmetadata: ", metadata, "\n")

    torch.save(
        {"config": convert_cfg_to_dict(config)},
        os.path.join(opt.save_dir, "config.pth"),
    )

    with open(os.path.join(opt.save_dir, "options.txt"), "w") as f:
        f.write(str(curriculum))

    from gmpi.eval.common import setup_model

    generator = setup_model(opt, config, metadata, mpi_xyz_input, mpi_xyz_only_z, vis_mesh=False, device=device)

    rgb_dir = os.path.join(opt.save_dir, opt.task, "rgb")
    depth_dir = os.path.join(opt.save_dir, opt.task, "depth")
    angle_dir = os.path.join(opt.save_dir, opt.task, "angle")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(angle_dir, exist_ok=True)

    global_cnt = 0
    mb_size = 1

    if opt.task == "consistency":
        n_view_per_z = 2
    else:
        n_view_per_z = 1

    print("\nsave_dir: ", rgb_dir, "\n")

    mean = torch.zeros((1, metadata["latent_dim"]), device=device)
    scale = torch.ones((1, metadata["latent_dim"]), device=device) * 0.01

    for start_i in tqdm.tqdm(range(0, opt.n_imgs, mb_size)):

        end_i = min(opt.n_imgs, start_i + mb_size)

        torch.manual_seed(start_i)
        z = torch.randn((end_i - start_i, metadata["latent_dim"]), device=device)
        # z = torch.normal(mean=mean, std=scale)

        # rgb: [B, H, W, 3]; depth: [B, H, W]
        img_list, depth_map_list, cam_angle_list = generate_img(
            metadata,
            generator,
            mpi_renderer,
            z,
            n_view_per_z,
            mpi_xyz_input,
            mpi_xyz_only_z=mpi_xyz_only_z,
            mpi_z_interpolation_ws=mpi_z_interpolation_ws,
            n_planes=opt.nplanes,
            disable_tqdm=True,
            light_render=None,
            stylegan2_sanity_check=bool(opt.stylegan2_sanity_check),
            truncation_psi=opt.truncation_psi,
            verbose=start_i == 0,
        )

        for i in range(end_i - start_i):

            if n_view_per_z == 1:
                tmp_rgb = img_list[0][i, ...]
                tmp_depth = depth_map_list[0][i, ...]
                tmp_angle = cam_angle_list[0][i, ...]

                tmp_idx = start_i + i

                Image.fromarray(tmp_rgb).save(os.path.join(rgb_dir, f"{tmp_idx:06d}.png"))

                with open(os.path.join(angle_dir, f"{tmp_idx:06d}.npy"), "wb") as f:
                    np.save(f, tmp_angle)

                if opt.save_depth == 1:
                    with open(os.path.join(depth_dir, f"{tmp_idx:06d}.npy"), "wb") as f:
                        np.save(f, tmp_depth)
            else:
                for j in range(n_view_per_z):

                    tmp_idx = i * n_view_per_z + j
                    tmp_rgb = img_list[0][tmp_idx, ...]
                    tmp_depth = depth_map_list[0][tmp_idx, ...]
                    tmp_angle = cam_angle_list[0][tmp_idx, ...]

                    Image.fromarray(tmp_rgb).save(os.path.join(rgb_dir, f"{start_i + i:06d}_{j}.png"))

                    with open(os.path.join(angle_dir, f"{start_i + i:06d}_{j}.npy"), "wb") as f:
                        np.save(f, tmp_angle)

                    if opt.save_depth == 1:
                        with open(os.path.join(depth_dir, f"{start_i + i:06d}_{j}.npy"), "wb") as f:
                            np.save(f, tmp_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_dir", type=str, default="imgs")
    parser.add_argument("--nplanes", type=int, default=32)
    parser.add_argument("--n_imgs", type=int, default=50000)
    parser.add_argument("--dataset", type=str, choices=["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"])
    parser.add_argument("--stylegan2_sanity_check", type=int, default=0)
    parser.add_argument("--save_depth", type=int, default=0)
    parser.add_argument("--truncation_psi", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="fid_kid", choices=["fid_kid", "consistency", "geometry"])
    parser.add_argument("--exp_config", type=str, default="./configs/pi_gan_with_mpi.yml")
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(opt)
