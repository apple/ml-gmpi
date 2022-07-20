import numpy as np
import torch
import torchvision

from gmpi.utils.cam_utils import gen_sphere_path
from gmpi.utils.torch_utils import normalize_vecs

EPS = 1e-8


class LightRenderer:
    """Modified from https://github.com/seasonSH/LiftedGAN/blob/main/models/renderers/renderer.py"""

    def __init__(
        self,
        *,
        sphere_center_z,
        sphere_r,
        ka_max=1.0,
        kd_max=0.0,
        n_grow_iters=1000,
        l_h_mean=0.0,
        l_h_std=0.2,
        l_v_mean=0.2,
        l_v_std=0.05,
        blur_ksize=9,
    ):
        self.ka_max = ka_max
        self.kd_max = kd_max
        self.n_grow_iters = n_grow_iters

        self.cur_ka = 0.0
        self.cur_kd = 0.0

        # light position on a sphere:
        # h: horizontal; v: vertical
        # NOTE: for v_mean = 0.2, we only sample light on the upper semisphere
        self.l_h_mean = l_h_mean
        self.l_h_std = l_h_std
        self.l_v_mean = l_v_mean
        self.l_v_std = l_v_std

        self.sphere_center = torch.FloatTensor(np.array([0, 0, sphere_center_z]))
        self.sphere_r = sphere_r

        self.blur_ksize = blur_ksize

        # NOTE: We smooth out depth image to make the lighting more realistic
        # Ref: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        blur_sigma = 0.3 * ((self.blur_ksize - 1) * 0.5 - 1) + 0.8
        self.blurrer_func = torchvision.transforms.GaussianBlur(
            kernel_size=(self.blur_ksize, self.blur_ksize), sigma=(blur_sigma, blur_sigma)
        )

        self.step = -1

    def get_normal(self, grid_3d, normalize=True):

        # [B, H, W, 3]
        center = grid_3d[:, 1:-1, 1:-1]
        # left, right, up, down = grid_3d[:,:-2,1:-1], grid_3d[:,2:,1:-1], grid_3d[:,1:-1,:-2], grid_3d[:,1:-1,2:]

        up, down, left, right = grid_3d[:, :-2, 1:-1], grid_3d[:, 2:, 1:-1], grid_3d[:, 1:-1, :-2], grid_3d[:, 1:-1, 2:]
        norm1 = torch.cross(up - center, left - center, dim=3)
        norm2 = torch.cross(left - center, down - center, dim=3)
        norm3 = torch.cross(down - center, right - center, dim=3)
        norm4 = torch.cross(right - center, up - center, dim=3)
        normal = norm1 + norm2 + norm3 + norm4

        # Zero Padding
        # zero = torch.FloatTensor([0,0,1]).to(depth.device)

        # [B, H, W, 3]
        normal = torch.nn.functional.pad(normal.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
        if normalize:
            normal = normal / (((normal**2).sum(3, keepdim=True)) ** 0.5 + EPS)

        # normal = normal.permute(1, 0, 2)

        return normal

    def compute_depth(self, mpi_alpha, plane_ds):

        bs = mpi_alpha.shape[0]

        # alpha-composition
        # [#mpi, #planes + 1, 1, img_h, img_w]
        alphas_shifted = torch.cat([torch.ones_like(mpi_alpha[:, :1, ...]), 1 - mpi_alpha + 1e-10], 1)
        # [#mpi, #planes, 1, img_h, img_w]
        weights = mpi_alpha * torch.cumprod(alphas_shifted, dim=1)[:, :-1, ...]

        # [#planes, 1, 1, 1]
        plane_ds = plane_ds.reshape((1, -1, 1, 1, 1))
        plane_ds = plane_ds.expand(bs, -1, -1, -1, -1)

        # [#mpi, 1, img_h, img_w]
        assert weights.ndim == plane_ds.ndim, f"{weights.shape}, {plane_ds.shape}"
        depth_out = torch.sum(weights * plane_ds, dim=1)

        return depth_out

    def compute_pcl(self, mpi_alpha, mpi_plane_dhws, mpi_tex_pix_xyz):
        # [#planes, 1]
        plane_ds = mpi_plane_dhws[:, :1].to(mpi_alpha.device)
        # [#mpi, 1, H, W] -> [#mpi, H, W]
        mpi_depth = self.compute_depth(mpi_alpha, plane_ds)

        # Use Gaussian blurring
        # [#mpi, 1, H, W] -> [#mpi, H, W]
        mpi_depth = self.blurrer_func(mpi_depth)[:, 0, ...]

        # create a point cloud for MPI
        # [1, H, W, 3]
        mpi_xyz_last_plane = mpi_tex_pix_xyz[-1:, :, :, :3]
        # get scale, [#mpi, H, W, 1]
        scale = mpi_depth.unsqueeze(-1) / (mpi_xyz_last_plane[..., 2:] + EPS)
        # [#mpi, H, W, 3]
        pcl = mpi_xyz_last_plane * scale

        return pcl

    def render(self, batch_mpi, mpi_plane_dhws, mpi_tex_pix_xyz):

        self.step += 1

        bs = batch_mpi.shape[0]

        # [bs, #planes, 3, H, W]
        mpi_rgb = batch_mpi[:, :, :3, ...]
        # [bs, #planes, 1, H, W]
        mpi_alpha = batch_mpi[:, :, 3:, ...]

        # [bs, H, W, 3]
        grid_3d = self.compute_pcl(mpi_alpha, mpi_plane_dhws, mpi_tex_pix_xyz)

        # for the world coordinate: +X right, +Y down, +Z forward
        batch_tf_c2w, batch_yaws, batch_pitches = gen_sphere_path(
            n_cams=bs,
            sphere_center=self.sphere_center,
            sphere_r=self.sphere_r,
            yaw_mean=self.l_h_mean,
            yaw_std=self.l_h_std,
            pitch_mean=self.l_v_mean,
            pitch_std=self.l_v_std,
            n_truncated_stds=2,
            flag_rnd=True,
            sample_method="truncated_gaussian",
            given_yaws=None,
            given_pitches=None,
        )

        # [B, 3]
        light_pos = batch_tf_c2w[:, :3, 3]
        if not isinstance(light_pos, torch.Tensor):
            light_pos = torch.FloatTensor(light_pos)
        light_pos = light_pos.to(mpi_alpha.device)

        # NOTE: the direction is towards the sphere center
        light_direction = torch.FloatTensor(self.sphere_center).reshape((1, 3)).to(mpi_alpha.device) - light_pos
        light_direction = normalize_vecs(light_direction)

        # light_in_direction = torch.FloatTensor([1/ np.sqrt(2), 0, 1/ np.sqrt(2)])

        ## shading
        # [B, H, W, 3]
        canon_normal = self.get_normal(grid_3d)
        # We assume both normal and light_direction are unit vectors.
        # [B, H, W]
        canon_diffuse_shading = (canon_normal * light_direction.view(-1, 1, 1, 3)).sum(3)
        # For Snell's law, we need to compute the cos(angle) between light_reflect_direction and normal
        canon_diffuse_shading = -1 * canon_diffuse_shading
        canon_diffuse_shading = canon_diffuse_shading.clamp(min=0)
        assert torch.min(canon_diffuse_shading) >= 0.0, f"{torch.min(canon_diffuse_shading)}"
        assert torch.max(canon_diffuse_shading) <= 1.0, f"{torch.max(canon_diffuse_shading)}"

        # [B, 1, 1, H, W]
        canon_diffuse_shading = canon_diffuse_shading.unsqueeze(1).unsqueeze(1)

        cur_ratio = min(1.0, self.step / self.n_grow_iters)

        self.cur_ka = cur_ratio * self.ka_max
        self.cur_kd = cur_ratio * self.kd_max

        cur_ka_torch = torch.ones((bs,), device=mpi_alpha.device) * self.cur_ka
        cur_kd_torch = torch.ones((bs,), device=mpi_alpha.device) * self.cur_kd

        # print("\nbatch_yaws: ", cur_ka, cur_kd, batch_yaws, batch_pitches, "\n")

        canon_shading = cur_ka_torch.view((bs, 1, 1, 1, 1)) + canon_diffuse_shading * cur_kd_torch.view(
            (bs, 1, 1, 1, 1)
        )

        # Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/diffuse-lambertian-shading
        new_mpi_rgb = mpi_rgb * canon_shading
        new_mpi_rgb = torch.clip(new_mpi_rgb, min=0.0, max=1.0)

        new_mpi = torch.cat((new_mpi_rgb, mpi_alpha), dim=2)

        return new_mpi
