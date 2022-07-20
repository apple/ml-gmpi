import argparse
import os
import random

import numpy as np
import torch
import trimesh
from torchvision.utils import save_image

import gmpi.curriculums as curriculums
from gmpi.core.mpi_renderer import MPIRenderer
from gmpi.train_helpers import modify_curriculums
from gmpi.utils import Config, get_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_mesh_mcubes(mpi_alpha, volume_min, volume_max):

    import mcubes

    # mpi_alpha: [#planes, H, W]
    mpi_alpha = mpi_alpha.permute(0, 2, 3, 1).cpu().numpy()[..., 0]
    print(
        "\nmpi_alpha: ",
        mpi_alpha.shape,
        np.min(mpi_alpha),
        np.max(mpi_alpha),
        np.mean(mpi_alpha),
        "\n",
    )

    mpi_alpha = mcubes.smooth(mpi_alpha)

    # [#verts, 3], [#faces, ]
    verts, faces = mcubes.marching_cubes(mpi_alpha, 0.01)

    # NOTE: we rotate the mesh so that it aligns with our coordinate system:
    # +X right, +Y down, +Z forward
    verts = verts[:, [2, 1, 0]]
    faces = faces[:, [1, 0, 2]]

    # transform to our MPI's volume.
    # assume we want to know p, s.t., verts / n_grid = (p - min) / (max - min)
    # --> p = verts / n_grid * (max - min) + min
    n_grid = mpi_alpha.shape[0]
    # range [0, 1]
    verts = verts / n_grid
    # translate to the center
    verts = verts * (volume_max - volume_min) + volume_min

    print("\nverts: ", verts.shape, faces.shape, "\n")
    print("\n", np.min(verts), np.max(verts), "\n")

    mesh = trimesh.base.Trimesh()
    mesh.vertices = verts
    mesh.faces = faces
    return mesh


def generate_mesh(
    gen,
    z,
    mpi_xyz_input,
    metadata,
    volume_min,
    volume_max,
    save_dir=None,
    n_all_planes=32,
    only_z=False,
    truncation_psi=1.0,
    stylegan2_sanity_check=False,
    plane_repeat=False,
    **kwargs,
):

    with torch.no_grad():
        # img, depth_map = generator.staged_forward(z, **kwargs)

        # assert n_mpi_actual_planes == mpi_xyz_input[4].shape[0], f"{n_mpi_actual_planes}, {mpi_xyz_input[4].shape[0]}"
        n_mpi_actual_planes = mpi_xyz_input[4].shape[0]

        mb_mpi_rgbas = []
        all_n_planes = mpi_xyz_input[4].shape[0]
        chunk_n_planes = 128
        for tmp_start_idx in range(0, all_n_planes, chunk_n_planes):
            tmp_end_idx = min(all_n_planes, tmp_start_idx + chunk_n_planes)
            tmp_mpi_xyz_input = {}
            for k in mpi_xyz_input:
                # [#planes, tex_h, tex_w, 3]
                tmp_mpi_xyz_input[k] = mpi_xyz_input[k][tmp_start_idx:tmp_end_idx, ...]

            tmp_mpi_rgbas = gen(
                z,
                None,
                tmp_mpi_xyz_input,
                only_z,
                tmp_end_idx - tmp_start_idx,
                truncation_psi=truncation_psi,
            )

            mb_mpi_rgbas.append(tmp_mpi_rgbas)

        mb_mpi_rgbas = torch.cat(mb_mpi_rgbas, dim=1)
        print("\nmb_mpi_rgbas: ", mb_mpi_rgbas.shape, "\n")

        single_mpi_rgb = mb_mpi_rgbas[:, 0, :3, ...]
        save_image(single_mpi_rgb, os.path.join(save_dir, f"ori_rgb.png"), nrow=1, normalize=False, padding=0)

        mpi_rgb = mb_mpi_rgbas[0, :, :3, ...]
        mpi_alpha = mb_mpi_rgbas[0, :, 3:, ...]

        if mpi_alpha.shape[2] != n_all_planes:
            # NOTE: for fast debug
            mpi_alpha = torch.nn.functional.interpolate(
                mpi_alpha, size=(n_all_planes, n_all_planes), mode="bilinear", align_corners=True
            )

        if stylegan2_sanity_check:
            # we make all alphas full while
            mpi_alpha = torch.ones_like(mpi_alpha)

        print("\nn_mpi_actual_planes: ", n_mpi_actual_planes, n_all_planes, "\n")

        if plane_repeat:
            n_cur_planes, _, h, w = mpi_alpha.shape
            n_repeat = int(np.ceil(n_mpi_actual_planes / n_cur_planes))
            # [#planes, 1, 1, H, W]
            mpi_alpha = mpi_alpha.unsqueeze(1).repeat(1, n_repeat, 1, 1, 1)
            mpi_alpha = mpi_alpha.reshape((n_cur_planes * n_repeat, 1, h, w))[:n_mpi_actual_planes, ...]

        n_grid_size = mpi_alpha.shape[-1]
        all_zeros_alpha = torch.zeros(
            (n_all_planes - n_mpi_actual_planes, 1, n_grid_size, n_grid_size), device=mpi_alpha.device
        )
        mpi_alpha = torch.cat((all_zeros_alpha, mpi_alpha), dim=0)
        print("\nmpi_alpha: ", mpi_alpha.shape, "\n")

        print("\ntex_size: ", metadata["tex_size"], "\n")
        mesh = extract_mesh_mcubes(mpi_alpha, volume_min, volume_max)

    return mesh


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

    mpi_xyz_only_z = True

    mpi_return_single_res_xyz = False

    from gmpi.eval.common import preprocess_for_extracting_mesh

    (
        mesh_nplanes,
        n_mpi_actual_planes,
        volume_min,
        volume_max,
        mpi_tex_pix_xyz_ph,
        mpi_tex_pix_normalized_xyz_ph,
    ) = preprocess_for_extracting_mesh(
        config, metadata, opt.nplanes, opt.tex_size, mpi_return_single_res_xyz, mpi_xyz_only_z, device
    )

    print("\n", metadata["h_stddev"], metadata["v_stddev"], "\n")

    mpi_renderer = MPIRenderer(
        n_mpi_planes=n_mpi_actual_planes,  # config.GMPI.MPI.n_gen_planes,
        plane_min_d=metadata["ray_start"],
        plane_max_d=metadata["ray_end"],
        plan_spatial_enlarge_factor=config.GMPI.MPI.CAM_SETUP.spatial_enlarge_factor,
        plane_distances_sample_method="uniform",
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
        mpi_xyz_input_ph = mpi_tex_pix_normalized_xyz_ph
    else:
        mpi_xyz_input = mpi_tex_pix_xyz
        mpi_xyz_input_ph = mpi_tex_pix_xyz_ph

    if mpi_xyz_only_z:
        print("\nmpi_xyz_input: ", mpi_xyz_input[4][:, 0, 0, 0], "\n")
    else:
        print("\nmpi_xyz_input: ", mpi_xyz_input[4][:, 0, 0, 2], "\n")

    print("\nconfig: ", config, "\n")
    print("\nmetadata: ", metadata, "\n")

    from gmpi.eval.common import setup_model

    generator = setup_model(opt, config, metadata, mpi_xyz_input_ph, mpi_xyz_only_z, vis_mesh=True, device=device)

    angle_for_horizontal = True

    if angle_for_horizontal:
        face_angles = np.linspace(0.5, -0.5, 100).tolist()
        face_angles = [a + metadata["h_mean"] for a in face_angles]
    else:
        face_angles = np.linspace(0.3, -0.3, 100).tolist()
        face_angles = [a + metadata["v_mean"] for a in face_angles]

    torch.manual_seed(opt.seed)
    print("\n", opt.seed, "\n")
    z = torch.randn((1, metadata["latent_dim"]), device=device)

    mesh = generate_mesh(
        generator,
        z,
        mpi_xyz_input,
        metadata,
        volume_min,
        volume_max,
        save_dir=opt.save_dir,
        n_all_planes=mesh_nplanes,
        only_z=mpi_xyz_only_z,
        truncation_psi=opt.truncation_psi,
        stylegan2_sanity_check=bool(opt.stylegan2_sanity_check),
        plane_repeat=(config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc == "none"),
    )

    mesh_f = os.path.join(opt.save_dir, f"mesh_{opt.truncation_psi}.ply")
    _ = mesh.export(mesh_f)

    print("\nout mesh_f: ", mesh_f, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="vis_mesh")
    parser.add_argument("--tex_size", type=int, default=512)
    parser.add_argument("--nplanes", type=int, default=32)
    parser.add_argument("--truncation_psi", type=float, default=1.0)
    # parser.add_argument("--n_imgs", type=int, default=50000)
    parser.add_argument("--dataset", type=str, choices=["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"])
    parser.add_argument("--stylegan2_sanity_check", type=int, default=0)
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
