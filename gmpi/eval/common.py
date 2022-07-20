import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from gmpi.core.mpi_renderer import MPIRenderer
from gmpi.train_helpers import STYLEGAN2_CFG_SPECS
from gmpi.utils import convert_cfg_to_dict

TRUNCATION_PSI = 1.0

PRETRAINED_CKPTS = {
    "FFHQ256": "stylegan2_pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl",
    "FFHQ512": "stylegan2_pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl",
    "FFHQ1024": "stylegan2_pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl",
    "AFHQCat": "stylegan2_pretrained/afhqcat.pkl",
    "MetFaces": "stylegan2_pretrained/metfaces.pkl",
}


def setup_model(opt, config, metadata, mpi_xyz_input, mpi_xyz_only_z, vis_mesh=False, device=torch.device("cpu")):

    n_g_out_channels = 4
    n_g_out_planes = opt.nplanes

    if "depth2alpha_n_z_bins" in config.GMPI.MPI and config.GMPI.MPI.depth2alpha_n_z_bins is not None:
        from gmpi.models.networks.networks_vanilla_depth2alpha import Generator as StyleGAN2Generator

        if config.GMPI.TRAIN.normalized_xyz_range == "01":
            depth2alpha_z_range = 1.0
        elif config.GMPI.TRAIN.normalized_xyz_range == "-11":
            depth2alpha_z_range = 2.0
        else:
            raise ValueError
    else:
        depth2alpha_z_range = 1.0

        if "depth2alpha_n_z_bins" not in config.GMPI.MPI:
            config.defrost()
            config.GMPI.MPI.depth2alpha_n_z_bins = None
            config.freeze()

        if config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc != "none":
            if config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func in ["learnable_param"]:
                from gmpi.models.networks.networks_pos_enc_learnable_param import Generator as StyleGAN2Generator

                n_g_out_planes = n_g_out_planes = config.GMPI.MPI.n_gen_planes
            else:
                from gmpi.models.networks.networks_cond_on_pos_enc import Generator as StyleGAN2Generator
        else:
            from gmpi.models.networks.networks_vanilla import Generator as StyleGAN2Generator

    synthesis_kwargs = convert_cfg_to_dict(config.GMPI.MODEL.STYLEGAN2.synthesis_kwargs)
    synthesis_kwargs_D = convert_cfg_to_dict(config.GMPI.MODEL.STYLEGAN2.synthesis_kwargs)
    # NOTE: ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/train.py#L178
    # if synthesis_kwargs["fmaps_for_channel_base"] != 1.0:
    #     assert config.GMPI.MODEL.STYLEGAN2.max_out_dim == 256, f"{config.GMPI.MODEL.STYLEGAN2.max_out_dim}"

    synthesis_kwargs["channel_base"] = int(
        STYLEGAN2_CFG_SPECS[str(config.GMPI.MODEL.STYLEGAN2.max_out_dim)]["fmaps"] * synthesis_kwargs["channel_base"]
    )

    synthesis_kwargs_D["channel_base"] = int(
        STYLEGAN2_CFG_SPECS[str(config.GMPI.MODEL.STYLEGAN2.max_out_dim_D)]["fmaps"]
        * synthesis_kwargs_D["channel_base"]
    )

    # For clamping, ref:
    # - Sec. D.1 Mixed-precision training of https://arxiv.org/abs/2006.06676
    # - https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/train.py#L333
    if config.GMPI.MODEL.STYLEGAN2.max_out_dim <= 128:
        synthesis_kwargs["num_fp16_res"] = 0
        synthesis_kwargs["conv_clamp"] = None
        synthesis_kwargs_D["num_fp16_res"] = 0
        synthesis_kwargs_D["conv_clamp"] = None
    else:
        synthesis_kwargs["conv_clamp"] = 256

        # Ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d4b2afe9c27e3c305b721bc886d2cb5229458eba/train.py#L181
        synthesis_kwargs["num_fp16_res"] = 4

    if vis_mesh:
        background_alpha_full = False
    else:
        background_alpha_full = config.GMPI.MODEL.STYLEGAN2.background_alpha_full

    generator = StyleGAN2Generator(
        z_dim=metadata["latent_dim"],  # config.GMPI.MODEL.STYLEGAN2.z_dim,
        c_dim=metadata["generator_label_dim"],  # config.GMPI.MODEL.STYLEGAN2.label_dim,
        w_dim=metadata["stylegan2_w_dim"],  # config.GMPI.MODEL.STYLEGAN2.w_dim,
        img_resolution=config.GMPI.MODEL.STYLEGAN2.max_out_dim,
        n_planes=n_g_out_planes,
        plane_channels=n_g_out_channels,
        pos_enc_multires=config.GMPI.MODEL.STYLEGAN2.pos_enc_multires,
        torgba_cond_on_pos_enc=config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc,
        torgba_cond_on_pos_enc_embed_func=config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func,
        torgba_sep_background=config.GMPI.MODEL.STYLEGAN2.torgba_sep_background,
        build_background_from_rgb=config.GMPI.MODEL.STYLEGAN2.build_background_from_rgb,
        build_background_from_rgb_ratio=config.GMPI.MODEL.STYLEGAN2.build_background_from_rgb_ratio,
        cond_on_pos_enc_only_alpha=config.GMPI.MODEL.STYLEGAN2.cond_on_pos_enc_only_alpha,
        gen_alpha_largest_res=config.GMPI.MODEL.STYLEGAN2.gen_alpha_largest_res,
        background_alpha_full=background_alpha_full,
        G_final_img_act=config.GMPI.MODEL.STYLEGAN2.G_final_img_act,
        mapping_kwargs=convert_cfg_to_dict(config.GMPI.MODEL.STYLEGAN2.mapping_kwargs),
        synthesis_kwargs=synthesis_kwargs,
        depth2alpha_n_z_bins=config.GMPI.MPI.depth2alpha_n_z_bins,
        depth2alpha_z_range=depth2alpha_z_range,
    ).to(device)

    import gmpi.models.torch_utils.misc as stylegan2_misc

    z = torch.empty([1, generator.z_dim], device=device)
    c = torch.empty([1, generator.c_dim], device=device)
    with torch.no_grad():
        _ = stylegan2_misc.print_module_summary(generator, [z, c, mpi_xyz_input, mpi_xyz_only_z, n_g_out_planes])

    if opt.stylegan2_sanity_check:
        import gmpi.models.legacy as stylegan2_legacy
        import gmpi.models.torch_utils.misc as stylegan2_misc
        from gmpi.models.dnnlib.util import open_url

        # pretrained_ckpt = "stylegan2_pretrained/metfaces.pkl"
        pretrained_ckpt = PRETRAINED_CKPTS[opt.dataset]

        print(f'Resuming from "{pretrained_ckpt}"')

        # NOTE: it requires torch_utils folder in PYTHONPATH
        with open_url(pretrained_ckpt) as f:
            resume_data = stylegan2_legacy.load_network_pkl(f)

        generator = generator.eval().requires_grad_(False)

        # NOTE: we resume G_ema to G
        for name, module in [("G_ema", generator)]:
            print(f"\n\nResume {name}\n")
            stylegan2_misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        # generator = generator.eval().requires_grad_(False)
    else:
        print(f"\nLoad weights from {opt.ckpt_path}\n")
        ema_file = opt.ckpt_path.split("generator")[0] + "ema.pth"
        print(f"\nLoad weights from {ema_file}\n")
        ema_state_dict = torch.load(ema_file)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema.load_state_dict(ema_state_dict)
        ema.copy_to(generator.parameters())

        # pretrained = torch.load(opt.ckpt_path, map_location=torch.device(device))
        # generator.load_state_dict(pretrained)

    generator = generator.eval()

    return generator


def preprocess_for_extracting_mesh(
    config, metadata, nplanes, tex_size, mpi_return_single_res_xyz, mpi_xyz_only_z, device
):

    # NOTE: we need this placeholder renderer to get the 3D volume of MPI.
    mpi_renderer_placeholder = MPIRenderer(
        n_mpi_planes=nplanes,
        plane_min_d=metadata["ray_start"],
        plane_max_d=metadata["ray_end"],
        plan_spatial_enlarge_factor=config.GMPI.MPI.CAM_SETUP.spatial_enlarge_factor,
        plane_distances_sample_method="uniform",  # config.GMPI.MPI.distance_sample_method,
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

    # [#planes, tex_h, tex_w, 4]
    mpi_tex_pix_xyz_ph, mpi_tex_pix_normalized_xyz_ph = mpi_renderer_placeholder.get_xyz(
        metadata["tex_size"],
        metadata["tex_size"],
        ret_single_res=mpi_return_single_res_xyz,
        only_z=mpi_xyz_only_z,
    )

    # [#planes, 3]
    mpi_dhws = mpi_renderer_placeholder.static_mpi_plane_dhws.cpu().numpy()
    mpi_z_size = np.abs(metadata["ray_end"] - metadata["ray_start"])
    mpi_hw_size = np.max(mpi_dhws[:-1, 1:])
    print("\nmpi_z_size: ", mpi_dhws.shape, mpi_dhws[0, :], mpi_z_size, mpi_hw_size, "\n")

    # +X right, +Y down, +Z forward
    half_x = np.max(np.abs(mpi_dhws[:-1, 2])) / 2
    half_y = np.max(np.abs(mpi_dhws[:-1, 1])) / 2
    x_min = -1 * half_x
    x_max = half_x
    y_min = -1 * half_y
    y_max = half_y
    z_max = metadata["ray_end"]
    z_min = z_max - mpi_hw_size  # NOTE: marching cube requires a cube
    volume_min = np.array([x_min, y_min, z_min]).reshape((-1, 3))
    volume_max = np.array([x_max, y_max, z_max]).reshape((-1, 3))
    volume_center = (volume_min + volume_max) / 2

    # NOTE: since marching cube requires cubic grid, we need to have H/W/D same length.
    mesh_nplanes = tex_size
    n_mpi_actual_planes = int(min(1.0, mpi_z_size / mpi_hw_size) * mesh_nplanes)
    print("\nmpi_dhws: ", mpi_dhws.shape, mpi_z_size, mpi_hw_size, "\n")
    print("\nn_mpi_actual_planes: ", n_mpi_actual_planes, "\n")

    return mesh_nplanes, n_mpi_actual_planes, volume_min, volume_max, mpi_tex_pix_xyz_ph, mpi_tex_pix_normalized_xyz_ph
