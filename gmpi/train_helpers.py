#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import datetime
import os

import torch
import torch.distributed as dist

import gmpi.curriculums as curriculums


# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/train.py#L154
# We change mbstd according to 1
STYLEGAN2_CFG_SPECS = {
    "auto": dict(
        ref_gpus=-1, kimg=25000, mb=-1, mbstd=1, fmaps=-1, lrate=-1, gamma=-1, ema=-1, ramp=0.05, map=2
    ),  # Populated dynamically based on resolution and GPU count.
    "stylegan2": dict(
        ref_gpus=8, kimg=25000, mb=32, mbstd=1, fmaps=1, lrate=0.002, gamma=10, ema=10, ramp=None, map=8
    ),  # Uses mixed-precision, unlike the original StyleGAN2.
    "256": dict(ref_gpus=8, kimg=25000, mb=64, mbstd=1, fmaps=0.5, lrate=0.0025, gamma=1, ema=20, ramp=None, map=8),
    "512": dict(ref_gpus=8, kimg=25000, mb=64, mbstd=1, fmaps=1, lrate=0.0025, gamma=0.5, ema=20, ramp=None, map=8),
    "1024": dict(ref_gpus=8, kimg=25000, mb=32, mbstd=1, fmaps=1, lrate=0.002, gamma=2, ema=10, ramp=None, map=8),
    "cifar": dict(
        ref_gpus=2, kimg=100000, mb=64, mbstd=1, fmaps=1, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2
    ),
}


def modify_curriculums(config, run_dataset=None, flag_eval=False):

    if run_dataset is not None:
        config.defrost()
        config.GMPI.TRAIN.dataset = run_dataset
        config.freeze()

    assert config.GMPI.TRAIN.dataset in [
        "FFHQ256",
        "FFHQ512",
        "FFHQ1024",
        "AFHQCat",
        "MetFaces",
    ], f"{config.GMPI.TRAIN.dataset}"

    config.defrost()

    if "FFHQ" in config.GMPI.TRAIN.dataset:
        res = int(config.GMPI.TRAIN.dataset[4:])
        curriculum_name = "FFHQ"
        config.GMPI.MPI.CAM_SETUP = config.GMPI.MPI.FOR_FFHQ
    elif "AFHQCat" in config.GMPI.TRAIN.dataset:
        res = 512
        curriculum_name = "AFHQCat"
        config.GMPI.MPI.CAM_SETUP = config.GMPI.MPI.FOR_AFHQCat
    elif "MetFaces" in config.GMPI.TRAIN.dataset:
        res = 1024
        curriculum_name = "MetFaces"
        config.GMPI.MPI.CAM_SETUP = config.GMPI.MPI.FOR_MetFaces
    else:
        raise ValueError

    config.GMPI.TRAIN.curriculum = curriculum_name

    config.GMPI.MODEL.STYLEGAN2.max_out_dim = res
    config.GMPI.MODEL.STYLEGAN2.max_out_dim_D = res
    config.GMPI.MODEL.pretrained = config.GMPI.MODEL.pretrained_ckpts[config.GMPI.TRAIN.dataset]
    config.GMPI.MODEL.pretrained_D = config.GMPI.MODEL.pretrained

    config.GMPI.MODEL.STYLEGAN2.discriminator.mbstd_group_size = STYLEGAN2_CFG_SPECS[
        str(config.GMPI.MODEL.STYLEGAN2.max_out_dim_D)
    ]["mbstd"]

    config.freeze()

    # Set dataset path right
    cur_curriculum = getattr(curriculums, curriculum_name)

    if config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func == "learnable_param":
        cur_curriculum[0].update(cur_curriculum["res_dict_learnable_param"][config.GMPI.MODEL.STYLEGAN2.max_out_dim_D])
    elif config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func in ["modulated_lrelu", "conv_lrelu", "none"]:
        cur_curriculum[0].update(cur_curriculum["res_dict"][config.GMPI.MODEL.STYLEGAN2.max_out_dim_D])
    else:
        raise ValueError(config.GMPI.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func)

    cur_curriculum["raw_img_size"] = config.GMPI.MODEL.STYLEGAN2.max_out_dim_D
    cur_curriculum["eval_img_size"] = config.GMPI.MODEL.STYLEGAN2.max_out_dim_D

    if curriculum_name == "FFHQ":
        cur_curriculum["dataset"] = "FFHQ"
        cur_curriculum["dataset_path"] = config.DATASET.FFHQ.TRAIN_DATAROOT.format(
            cur_curriculum["raw_img_size"], cur_curriculum["raw_img_size"]
        )
        cur_curriculum["pose_data_path"] = config.DATASET.FFHQ.POSE_DATAROOT.format(cur_curriculum["raw_img_size"])
    elif curriculum_name == "AFHQCat":
        cur_curriculum["dataset"] = "AFHQCat"
        cur_curriculum["dataset_path"] = config.DATASET.AFHQCat.TRAIN_DATAROOT
        cur_curriculum["pose_data_path"] = config.DATASET.AFHQCat.POSE_DATAROOT
    elif curriculum_name == "MetFaces":
        cur_curriculum["dataset"] = "MetFaces"
        cur_curriculum["dataset_path"] = config.DATASET.MetFaces.TRAIN_DATAROOT
        cur_curriculum["pose_data_path"] = config.DATASET.MetFaces.POSE_DATAROOT
    else:
        raise ValueError

    setattr(curriculums, curriculum_name, cur_curriculum)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=7200))


def cleanup():
    dist.destroy_process_group()


def z_sampler(shape, device, dist):
    if dist == "gaussian":
        z = torch.randn(shape, device=device)
    elif dist == "uniform":
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def find_worst_view_per_z(
    *,
    z,
    generator_ddp,
    discriminator_ddp,
    mpi_renderer,
    config,
    metadata,
    stylegan2_mpi_xyz_input,
    enable_mapping_grad,
    enable_syn_feat_net_grad,
    cur_h_stddev,
    cur_v_stddev,
    alpha,
    xyz_coords_only_z,
    n_planes,
    truncation_psi,
):

    chosen_cam_pos = []

    bs = z.shape[0]

    if metadata["tex_size"] == 1024:
        mb = 1
    elif metadata["tex_size"] == 512:
        mb = 2
    else:
        mb = 4

    for start_batch_idx in range(0, bs, mb):

        end_batch_idx = min(bs, start_batch_idx + mb)
        subset_z = z[start_batch_idx:end_batch_idx]

        split_batch_size = end_batch_idx - start_batch_idx

        raw_batch_mpi_rgbas = generator_ddp(
            z=subset_z,
            c=None,
            mpi_xyz_coords=stylegan2_mpi_xyz_input,
            xyz_coords_only_z=xyz_coords_only_z,
            n_planes=n_planes,
            enable_mapping_grad=enable_mapping_grad,
            enable_syn_feat_net_grad=enable_syn_feat_net_grad,
            truncation_psi=truncation_psi,
        )

        # [B, #planes, 4, H, W]
        tmp_bs, tmp_n_planes, _, tmp_h, tmp_w = raw_batch_mpi_rgbas.shape
        batch_mpi_rgbas = raw_batch_mpi_rgbas.unsqueeze(1).expand(
            -1, config.GMPI.TRAIN.n_view_per_z_in_train, -1, -1, -1, -1
        )
        batch_mpi_rgbas = batch_mpi_rgbas.reshape(
            (tmp_bs * config.GMPI.TRAIN.n_view_per_z_in_train, tmp_n_planes, 4, tmp_h, tmp_w)
        )

        gen_imgs, _, gen_c2w_mats, gen_positions = mpi_renderer.render(
            batch_mpi_rgbas,
            metadata["img_size"],
            metadata["img_size"],
            horizontal_std=cur_h_stddev,
            vertical_std=cur_v_stddev,
        )

        if config.GMPI.TRAIN.D_cond_pose_dim == 9:
            gen_w2c_mats = torch.inverse(gen_c2w_mats[:, :3, :3])
        elif config.GMPI.TRAIN.D_cond_pose_dim == 16:
            gen_w2c_mats = torch.inverse(gen_c2w_mats)
        else:
            raise ValueError
        flat_gen_w2c_mats = gen_w2c_mats.reshape((split_batch_size * config.GMPI.TRAIN.n_view_per_z_in_train, -1))

        if config.GMPI.TRAIN.D_cond_on_pose:
            g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, flat_gen_w2c_mats, **metadata)
        else:
            g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, None, **metadata)

        assert (config.GMPI.TRAIN.n_view_per_z_in_train > 1) and (config.GMPI.TRAIN.G_select_worse_view)

        # NOTE: we need to choose the worst-scored view
        # g_pred: [B x #views_per_z, 1]
        # softplus: https://github.com/pfnet-research/sngan_projection/issues/18#issuecomment-392683263
        # [mb, #view_per_z, 1]
        tmp_g_preds = -1 * g_preds.reshape((-1, config.GMPI.TRAIN.n_view_per_z_in_train, 1))
        # tmp_hard_examples = -1 * torch.topk(tmp_g_preds, 1, dim=1).values
        # g_preds = tmp_hard_examples.reshape((-1, 1))

        # [mb, 1, 1]
        tmp_hard_idxs = torch.topk(tmp_g_preds, 1, dim=1, largest=True, sorted=True).indices
        # [mb, 1, 2]
        tmp_hard_idxs = tmp_hard_idxs.expand(-1, -1, 2)

        # [mb, #view_per_z, 2]
        tmp_gen_positions = gen_positions.reshape((-1, config.GMPI.TRAIN.n_view_per_z_in_train, 2))

        # [mb, 1, 2] -> [mb, 2]
        tmp_chosen_cam_pos = torch.gather(tmp_gen_positions, dim=1, index=tmp_hard_idxs)[:, 0, :]

        chosen_cam_pos.append(tmp_chosen_cam_pos)

    chosen_cam_pos = torch.cat(chosen_cam_pos, dim=0)

    chosen_cam_pitches = chosen_cam_pos[:, :1]
    chosen_cam_yaws = chosen_cam_pos[:, 1:]

    return chosen_cam_yaws, chosen_cam_pitches, z
