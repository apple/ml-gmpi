import math
from typing import List

import torch
import torch.nn.functional as F


def range_01_to_pm1(img):
    """Convert from range [0, 1] to [-1, 1]"""
    assert isinstance(img, torch.Tensor)
    assert torch.min(img) >= 0.0 and torch.max(img) <= 1.0, f"{torch.min(img)}, {torch.max(img)}"
    new_img = (img - 0.5) * 2
    return new_img


def range_pm1_to_01(img):
    """Convert from range [-1, 1] to [0, 1]"""
    assert isinstance(img, torch.Tensor)
    assert torch.min(img) >= -1.0 and torch.max(img) <= 1.0, f"{torch.min(img)}, {torch.max(img)}"
    new_img = (img + 1) / 2
    return new_img


# https://github.com/pytorch/vision/blob/de3e9091aba94059b4caa79e2c31b78e82add880/torchvision/transforms/functional.py#L894
def get_inverse_affine_matrix(
    center: List[float],
    angle: float,
    translate: List[float],
    scale: float,
    shear: List[float],
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


# https://github.com/pytorch/vision/blob/de3e9091aba94059b4caa79e2c31b78e82add880/torchvision/transforms/functional_tensor.py#L660
def gen_affine_grid(
    theta,
    w: int,
    h: int,
    ow: int,
    oh: int,
):
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)
    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


# ----------------------------------------------------------------------
# classical RGB/depth gradient

# modified from https://github.com/kornia/kornia/blob/3606cf9c3d1eb3aabd65ca36a0e7cb98944c01ba/kornia/filters/filter.py#L32


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def filter2D(in_tensor, kernel):
    """
    in_tensor: [B, in_C, H, W]
    kernel: [B, kH, kW]
    """
    b, c, h, w = in_tensor.shape
    tmp_kernel = kernel.unsqueeze(1).to(in_tensor)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1).contiguous()
    # print("tmp_kernel: ", tmp_kernel.shape)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input_pad = F.pad(in_tensor, padding_shape, mode="reflect")
    # print("input_pad: ", input_pad.shape)

    out_tensor = F.conv2d(input_pad, tmp_kernel, padding=0, stride=1)
    # print("out_tensor: ", out_tensor.shape)

    return out_tensor


Sobel_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sobel_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


CLASSICAL_KERNELS = {
    "Sobel": {"x": [Sobel_X, 1], "y": [Sobel_Y, 1]},
}


def compute_img_grad(img, kernel_name="Sobel"):
    kernel_x = CLASSICAL_KERNELS[kernel_name]["x"][0]
    kernel_y = CLASSICAL_KERNELS[kernel_name]["y"][0]
    classic_grad_x = torch.abs(filter2D(img, kernel_x.unsqueeze(0).to(img.device)))
    classic_grad_y = torch.abs(filter2D(img, kernel_y.unsqueeze(0).to(img.device)))
    # [B, 1, H, W]
    img_grad = (classic_grad_x + classic_grad_y) / 2
    return img_grad


def edge_aware_smooth_loss(rgb, depth, e_min=0.05, g_min=0.01):
    """
    Ref: Sec. 3.4 of https://arxiv.org/abs/2004.11364

    e_min: for RGB. When Grad(RGB) > e_min * max(Grad(RGB)), we treat it as an edge.
    g_min: for depth. When Grad(depth) > g_min, we treat it as an edge.

    rgb: [B, 3, H, W]
    depth: [B, 3, H, W]
    """

    eps = 1e-8

    rgb_grad = compute_img_grad(rgb, kernel_name="Sobel")
    depth_grad = compute_img_grad(depth, kernel_name="Sobel")

    edge_rgb_threshold = torch.max(rgb_grad) * e_min
    rgb_edge = rgb_grad / (edge_rgb_threshold + eps)
    # [B, 1, H, W]
    rgb_edge = torch.minimum(rgb_edge, torch.ones(rgb_edge.shape, device=rgb_edge.device))

    # edge_depth_threshold = torch.max(depth_grad) * g_min
    edge_depth_threshold = g_min
    # [B, 1, H, W]
    depth_edge = torch.maximum(
        depth_grad - edge_depth_threshold, torch.zeros(depth_grad.shape, device=depth_grad.device)
    )

    # We ignore areas where RGB shows edges. Namely, we want depth to be as smooth as possible when RGB has no edges.
    mask = 1 - rgb_edge

    # loss = torch.sum(depth_edge * mask) / (torch.sum(mask > 0) + eps)
    loss = torch.mean(depth_edge * mask)

    return loss
