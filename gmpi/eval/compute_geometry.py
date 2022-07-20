import argparse
import glob
import json
import os

import joblib
import numpy as np
import scipy.io as sio
import torch
import tqdm

SPHERE_CENTER = 1.0
SPHERE_R = 1.0
SPHERE_CENTER_VEC = torch.FloatTensor([0, 0, SPHERE_CENTER])


def normalize_vec(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    norm_vec = (vec - mean) / (std + 1e-8)
    return norm_vec


def compute_depth_err(alinged_depth_f, pred_depth_f, pred_mask_f):

    depth = np.load(alinged_depth_f)

    pred_depth = np.load(pred_depth_f)
    pred_mask = np.load(pred_mask_f)

    pred_mask[depth < 1e-8] = 0

    # [#pixels, ]
    valid_rows, valid_cols = np.where(pred_mask == 1)

    pred_depth_pixs = pred_depth[valid_rows, valid_cols]
    depth_pixs = depth[valid_rows, valid_cols]

    norm_pred_depth = normalize_vec(pred_depth_pixs)
    norm_depth = normalize_vec(depth_pixs)

    err = np.mean(np.square(norm_pred_depth - norm_depth))

    return err


def compute_angle_err(render_angle_f, pred_coeff_f):
    """
    NOTE: Deep3DFace use angles for rotation in the order of Rx, Ry, Rz (Rx: rotation along x-axis).
    - They define coordinate system as +X right, +Y up, +Z backward.
    - We use +X right, +Y down, +Z up
    Therefore:
    - our -1 * yaw corresponds to their 2nd angle
    - our pitch corresponds to their 1st angle
    - our -1 roll corresponds to their 3rd angle. Though our roll is always zero.
    """
    pred_coeffs = sio.loadmat(pred_coeff_f)
    # [1, 3]
    pred_angles = pred_coeffs["angle"][0, :]

    # [2,], [pitch, yaw]
    render_angles = np.load(render_angle_f)
    # [3, ]
    correspond_angles = np.array([render_angles[0], -1 * render_angles[1], 0])

    err = np.mean(np.square(pred_angles - correspond_angles))

    return err


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--angle_err", type=int, required=True)
    args = parser.parse_args()

    all_fs = sorted(list(glob.glob(os.path.join(args.geo_dir, f"rgb/detections/0*.txt"))))

    depth_err_dict = {}
    angle_err_dict = {}

    for tmp_f in tqdm.tqdm(all_fs):

        i = int(os.path.basename(tmp_f).split(".")[0])

        tmp_aligned_depth_f = os.path.join(args.geo_dir, f"recon/aligned_depth/{i:06d}.npy")
        tmp_pred_depth_f = os.path.join(args.geo_dir, f"recon/pred_depth/{i:06d}.npy")
        tmp_pred_mask_f = os.path.join(args.geo_dir, f"recon/pred_mask/{i:06d}.npy")
        tmp_pred_coeff_f = os.path.join(args.geo_dir, f"recon/coeffs/{i:06d}.mat")
        tmp_render_angle_f = os.path.join(args.geo_dir, f"angle/{i:06d}.npy")

        tmp_depth_err = compute_depth_err(tmp_aligned_depth_f, tmp_pred_depth_f, tmp_pred_mask_f)

        if args.angle_err == 1:
            tmp_angle_err = compute_angle_err(tmp_render_angle_f, tmp_pred_coeff_f)
        else:
            tmp_angle_err = 0

        depth_err_dict[i] = tmp_depth_err
        angle_err_dict[i] = tmp_angle_err

    depth_err = np.mean(list(depth_err_dict.values()))
    depth_err_std = np.std(list(depth_err_dict.values()))
    angle_err = np.mean(list(angle_err_dict.values()))
    angle_err_std = np.std(list(angle_err_dict.values()))
    print("\n", args.geo_dir, "\n")
    print("\ndepth: ", depth_err, depth_err_std, "\n")
    print("\nangle: ", angle_err, angle_err_std, "\n")

    with open(os.path.join(args.geo_dir, "depth_err.pt"), "wb") as f:
        joblib.dump(depth_err_dict, f, compress="lz4")

    with open(os.path.join(args.geo_dir, "angle_err.pt"), "wb") as f:
        joblib.dump(angle_err_dict, f, compress="lz4")

    save_dict = {
        "depth": str(depth_err),
        "depth_std": str(depth_err_std),
        "angle": str(angle_err),
        "angle_std": str(angle_err_std),
    }

    with open(os.path.join(args.geo_dir, "aggregated.json"), "w") as f:
        json.dump(save_dict, f)
