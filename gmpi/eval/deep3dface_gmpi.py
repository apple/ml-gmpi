"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os

import numpy as np
import torch
import tqdm
from models import create_model
from options.test_options import TestOptions
from PIL import Image
from util.load_mats import load_lm3d
from util.preprocess import align_img
from util.visualizer import MyVisualizer


def get_data_path(img_root, depth_root, detect_root):

    if os.path.exists(os.path.join(detect_root, "fail_list.txt")):
        with open(os.path.join(detect_root, "fail_list.txt"), "r") as f:
            fail_list = [_.strip() for _ in f.readlines()]
    else:
        fail_list = []

    print("\nfail_list: ", fail_list, "\n")

    all_im_path = [
        os.path.join(img_root, i) for i in sorted(os.listdir(img_root)) if i.endswith("png") or i.endswith("jpg")
    ]
    # filter out failing cases
    im_path = []
    for elem in all_im_path:
        if os.path.basename(elem) not in fail_list:
            im_path.append(elem)
    print(f"\nFind {len(im_path)} valid images from {len(all_im_path)} images.\n")

    lm_path = [i.replace("png", "txt").replace("jpg", "txt") for i in im_path]
    lm_path = [os.path.join(detect_root, os.path.basename(i)) for i in lm_path]

    depth_path = [i.replace("png", "npy").replace("jpg", "npy") for i in im_path]
    depth_path = [os.path.join(depth_root, os.path.basename(i)) for i in depth_path]

    return im_path, lm_path, depth_path


def read_data(im_path, depth_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    # [H, W]
    depth = Image.fromarray(np.load(depth_path)[..., 0])
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _, depth = align_img(im, lm, lm3d_std, depth=depth)
    if to_tensor:
        im = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    depth = np.array(depth)
    return im, lm, depth


def main(rank, opt, img_root, depth_root, detect_root):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path, depth_path = get_data_path(img_root, depth_root, detect_root)
    lm3d_std = load_lm3d(opt.bfm_folder)

    base_dir = os.path.dirname(img_root)
    # save_dir = os.path.join(visualizer.img_dir, os.path.basename(img_root), 'epoch_%s_%06d'%(opt.epoch, 0))
    save_dir = os.path.join(base_dir, "recon")
    os.makedirs(save_dir, exist_ok=True)

    pred_mask_dir = os.path.join(save_dir, "pred_mask")
    pred_depth_dir = os.path.join(save_dir, "pred_depth")
    aligned_rgb_dir = os.path.join(save_dir, "aligned_rgb")
    aligned_depth_dir = os.path.join(save_dir, "aligned_depth")
    coeffs_dir = os.path.join(save_dir, "coeffs")
    for tmp in [pred_mask_dir, pred_depth_dir, aligned_rgb_dir, aligned_depth_dir, coeffs_dir]:
        os.makedirs(tmp, exist_ok=True)

    BATCH_N = 20

    for i in tqdm.tqdm(range(len(im_path))):
        # print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace(".png", "").replace(".jpg", "")
        # if not os.path.isfile(lm_path[i]):
        #     continue
        assert os.path.isfile(lm_path[i]), lm_path[i]
        im_tensor, lm_tensor, depth_np = read_data(im_path[i], depth_path[i], lm_path[i], lm3d_std)
        data = {"imgs": im_tensor, "lms": lm_tensor}
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        # [B, 1, H, W], H = W = 224
        pred_mask = model.pred_mask
        # [B, 1, H, W], H = W = 224
        pred_depth = model.pred_depth

        # print("\npred_mask: ", pred_mask.shape, pred_depth.shape, depth_np.shape, depth_np.dtype, "\n")

        # [B, 3, H, W], range [0, 1]
        aligned_im = im_tensor.permute(0, 2, 3, 1).cpu().numpy()[0, ...]
        aligned_im = (aligned_im * 255).astype(np.uint8)
        pred_mask = pred_mask.cpu().numpy()[0, 0, ...]
        pred_depth = pred_depth.cpu().numpy()[0, 0, ...]

        Image.fromarray(aligned_im).save(os.path.join(aligned_rgb_dir, f"{img_name}.png"))

        with open(os.path.join(pred_mask_dir, f"{img_name}.npy"), "wb") as f:
            np.save(f, pred_mask)

        with open(os.path.join(pred_depth_dir, f"{img_name}.npy"), "wb") as f:
            np.save(f, pred_depth)

        with open(os.path.join(aligned_depth_dir, f"{img_name}.npy"), "wb") as f:
            np.save(f, depth_np)

        model.save_coeff(os.path.join(coeffs_dir, f"{img_name}.mat"))  # save predicted coefficients


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options

    main(0, opt, opt.gmpi_img_root, opt.gmpi_depth_root, opt.gmpi_detect_root)
