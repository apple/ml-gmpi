import argparse
import os

import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.utils import save_image

import gmpi.curriculums as curriculums
from gmpi.core.mpi_renderer import MPIRenderer
from gmpi.train_helpers import modify_curriculums
from gmpi.utils import Config, get_config
from gmpi.utils.io_utils import images_to_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_img(
    *,
    device,
    face_angles,
    generator,
    z,
    mpi_xyz_input,
    metadata,
    horizontal_cam_move=True,
    save_dir=None,
    mpi_xyz_only_z=False,
    z_interpolation_ws=None,
    n_planes=32,
    light_render=None,
    disable_tqdm=False,
    truncation_psi=1.0,
    render_single_image=True,
    chunk_n_planes=-1,
    **kwargs,
):

    with torch.no_grad():

        mb_mpi_rgbas = generator(
            z,
            None,
            mpi_xyz_input,
            mpi_xyz_only_z,
            n_planes,
            z_interpolation_ws=z_interpolation_ws,
            truncation_psi=truncation_psi,
        )

        mb_mpi_rgbas = []
        all_n_planes = mpi_xyz_input[4].shape[0]

        if chunk_n_planes == -1:
            chunk_n_planes = all_n_planes + 1

        for tmp_start_idx in tqdm.tqdm(range(0, all_n_planes, chunk_n_planes)):
            tmp_end_idx = min(all_n_planes, tmp_start_idx + chunk_n_planes)
            tmp_mpi_xyz_input = {}
            for k in mpi_xyz_input:
                # [#planes, tex_h, tex_w, 3]
                tmp_mpi_xyz_input[k] = mpi_xyz_input[k][tmp_start_idx:tmp_end_idx, ...]

            tmp_mpi_rgbas = generator(
                z,
                None,
                tmp_mpi_xyz_input,
                mpi_xyz_only_z,
                tmp_end_idx - tmp_start_idx,
                z_interpolation_ws=z_interpolation_ws,
                truncation_psi=truncation_psi,
            )

            mb_mpi_rgbas.append(tmp_mpi_rgbas)

        mb_mpi_rgbas = torch.cat(mb_mpi_rgbas, dim=1)
        print("\nmb_mpi_rgbas: ", mb_mpi_rgbas.shape, "\n")

        print("\ntruncation_psi: ", truncation_psi, "\n")

        torch.cuda.empty_cache()

        mpi_1st_rgb = mb_mpi_rgbas[:, 0, :3, ...]

        # [#planes, 3, H, W]
        mpi_rgb = mb_mpi_rgbas[0, :, :3, ...]
        # [#planes, 1, H, W]
        mpi_alpha = mb_mpi_rgbas[0, :, 3:, ...]

        tensor_img_list = []
        img_list = []
        depth_img_list = []

        for i, tmp_angle in tqdm.tqdm(enumerate(face_angles), total=len(face_angles), disable=disable_tqdm):

            if render_single_image:
                metadata["h_mean"] = tmp_angle[0]
                metadata["v_mean"] = tmp_angle[1]
                print(f"\nrendering angles horizontal {tmp_angle[0]}; vertical {tmp_angle[1]}\n")
            else:
                if horizontal_cam_move:
                    metadata["h_mean"] = tmp_angle
                else:
                    metadata["v_mean"] = tmp_angle

            img, depth_map, _, _ = mpi_renderer.render(
                mb_mpi_rgbas,
                metadata["img_size"],
                metadata["img_size"],
                horizontal_mean=metadata["h_mean"],
                horizontal_std=metadata["h_stddev"],
                vertical_mean=metadata["v_mean"],
                vertical_std=metadata["v_stddev"],
                assert_not_out_of_last_plane=True,
            )

            tensor_img = img.detach()
            img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            img = (img + 1) / 2.0
            img = (img * 255).astype(np.uint8)

            depth_map = depth_map.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            depth_map = (depth_map - metadata["ray_start"]) / (metadata["ray_end"] - metadata["ray_start"])
            depth_map = np.clip(depth_map, 0, 1)
            depth_map = (depth_map[..., None] * 255).astype(np.uint8)

            tensor_img_list.append(tensor_img)
            img_list.append(img)
            depth_img_list.append(depth_map)

    return img_list, tensor_img_list, depth_img_list, mpi_rgb, mpi_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--seeds", nargs="+", default=[0])
    parser.add_argument("--render_single_image", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./imgs")
    parser.add_argument("--nplanes", type=int, default=32)
    parser.add_argument("--truncation_psi", type=float, default=1.0)
    parser.add_argument("--horizontal_cam_move", type=int, default=1)
    parser.add_argument("--chunk_n_planes", type=int, default=-1)
    parser.add_argument("--exp_config", type=str, default="./configs/gmpi.yml")
    opt = parser.parse_args()

    print("\nos.path.splitext(opt.exp_config): ", os.path.splitext(opt.exp_config)[1], "\n")
    if os.path.splitext(opt.exp_config)[1] in [".yml", ".yaml"]:
        config = get_config(opt.exp_config, None)
    elif os.path.splitext(opt.exp_config)[1] == ".pth":
        config = Config(init_dict=torch.load(opt.exp_config, map_location="cpu")["config"])
    else:
        raise ValueError

    if "depth2alpha_n_z_bins" not in config.GMPI.MPI:
        config.defrost()
        config.GMPI.MPI.depth2alpha_n_z_bins = None
        config.freeze()

    mpi_xyz_only_z = False

    modify_curriculums(config, flag_eval=True)

    curriculum = getattr(curriculums, config.GMPI.TRAIN.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    mpi_renderer = MPIRenderer(
        n_mpi_planes=opt.nplanes,  # config.GMPI.MPI.n_gen_planes,
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

    # NOTE: we need to manually control the angle
    # we must set stddev to zero after setting up MPI renderer
    metadata["h_stddev"] = 0.0
    metadata["v_stddev"] = 0.0

    print("\nconfig: ", config, "\n")
    print("\nmetadata: ", metadata, "\n")

    from gmpi.eval.common import setup_model

    opt.stylegan2_sanity_check = False
    generator = setup_model(opt, config, metadata, mpi_xyz_input, mpi_xyz_only_z, vis_mesh=False, device=device)

    horizontal_cam_move = bool(opt.horizontal_cam_move)

    if bool(opt.render_single_image):
        face_angles = [[np.random.uniform(0.5, -0.5), np.random.uniform(0.3, -0.3)]]
    else:
        if horizontal_cam_move:
            face_angles = np.linspace(0.5, -0.5, 100).tolist()
            face_angles = [a + metadata["h_mean"] for a in face_angles]
        else:
            face_angles = np.linspace(0.3, -0.3, 100).tolist()
            face_angles = [a + metadata["v_mean"] for a in face_angles]

    for seed in tqdm.tqdm(opt.seeds):

        torch.manual_seed(seed)
        z = torch.randn((1, metadata["latent_dim"]), device=device)

        img_list, tensor_img_list, depth_img_list, mpi_rgb, mpi_alpha = generate_img(
            device=device,
            face_angles=face_angles,
            generator=generator,
            z=z,
            mpi_xyz_input=mpi_xyz_input,
            metadata=metadata,
            use_normalized_xyz=config.GMPI.TRAIN.use_normalized_xyz,
            horizontal_cam_move=horizontal_cam_move,
            mpi_xyz_only_z=mpi_xyz_only_z,
            z_interpolation_ws=mpi_z_interpolation_ws,
            n_planes=opt.nplanes,
            truncation_psi=opt.truncation_psi,
            save_dir=opt.output_dir,
            render_single_image=bool(opt.render_single_image),
            chunk_n_planes=opt.chunk_n_planes,
        )

        if bool(opt.render_single_image):
            Image.fromarray(img_list[0]).save(os.path.join(opt.output_dir, f"rendered.png"))
        else:
            if horizontal_cam_move:
                angle_type = "horizontal"
            else:
                angle_type = "vertical"
            # The following may take some time (several minutes) for high_res texture, e.g., 1024.
            images_to_video(
                images=img_list,
                output_dir=opt.output_dir,
                video_name=f"video_rgb_{angle_type}",
                fps=8,
                quality=5,
                disable_tqdm=True,
            )

            images_to_video(
                images=depth_img_list,
                output_dir=opt.output_dir,
                video_name=f"video_depth_{angle_type}",
                fps=8,
                quality=5,
                disable_tqdm=True,
            )

        # The following may take some time (several minutes) for high_res texture, e.g., 1024.
        save_image(
            mpi_alpha,
            os.path.join(opt.output_dir, f"mpi_alpha.png"),
            nrow=8,
            normalize=True,
        )

        save_image(
            mpi_rgb,
            os.path.join(opt.output_dir, f"mpi_rgb.png"),
            nrow=8,
            normalize=False,
        )

        save_image(
            torch.cat((mpi_rgb, mpi_alpha), dim=1),
            os.path.join(opt.output_dir, f"mpi_rgba.png"),
            nrow=8,
            normalize=False,
        )
