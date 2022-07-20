"""Datasets"""

import json
import os
import random
import zipfile

import numpy as np
import PIL
import pyspng
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from gmpi.utils.cam_utils import (
    compute_pitch_yaw_from_w2c_mat,
    compute_w2c_mat_from_estimated_pose_afhq,
    compute_w2c_mat_from_estimated_pose_ffhq,
)


class FFHQ(Dataset):
    """FFHQ Dataset"""

    def __init__(
        self,
        dataset_path,
        raw_img_size,
        img_size,
        pose_data_path,
        sphere_center,
        sphere_r=1.0,
        flat_pose_dim=9,
        **kwargs,
    ):
        super().__init__()

        if os.path.exists(os.path.join(pose_data_path, "fail_list.txt")):
            with open(os.path.join(pose_data_path, "fail_list.txt"), "r") as f:
                fail_list = [_.strip() for _ in f.readlines()]
        else:
            fail_list = []

        self.zipf = dataset_path
        self.zip_obj = None

        all_f_list = zipfile.ZipFile(self.zipf).namelist()

        PIL.Image.init()
        all_img_f_list = [_ for _ in all_f_list if self.get_file_ext(_) in PIL.Image.EXTENSION]
        all_im_path = sorted(all_img_f_list)
        print("\nsorted_f_list: ", len(all_im_path), all_im_path[:5], "\n")

        # filter out failing cases
        im_path = []
        for elem in all_im_path:
            if elem not in fail_list:
                im_path.append(elem)

        lm_path = [i.replace("png", "mat") for i in im_path]
        pose_data = [os.path.join(pose_data_path, i) for i in lm_path]

        self.data = list(zip(im_path, pose_data))

        print(f"\nFind {len(self.data)} valid images from {len(all_im_path)} images.\n")

        self.sphere_center = sphere_center
        self.sphere_center_vec = torch.FloatTensor([0, 0, sphere_center])
        self.sphere_r = sphere_r
        self.flat_pose_dim = flat_pose_dim

        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.raw_img_size = raw_img_size
        self.img_size = img_size

        if self.raw_img_size != self.img_size:
            # transforms.InterpolationMode.LANCZOS
            transform_list = [transforms.Resize((img_size, img_size), interpolation=PIL.Image.LANCZOS)]
        else:
            transform_list = []

        transform_list.extend(
            [
                # transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Resize((img_size, img_size), interpolation=0),
            ]
        )
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.zip_obj is None:
            self.zip_obj = zipfile.ZipFile(self.zipf)

        img_f, pose_f = self.data[index]

        with self.zip_obj.open(img_f, "r") as f:
            # X = PIL.Image.open(f)
            X = pyspng.load(f.read())
        assert X.shape[0] == self.raw_img_size, f"{X.shape}, {self.raw_img_size}"
        assert X.shape[1] == self.raw_img_size, f"{X.shape}, {self.raw_img_size}"

        X = self.transform(Image.fromarray(X))

        assert X.shape[1] == self.img_size, f"{X.shape}, {self.img_size}"
        assert X.shape[2] == self.img_size, f"{X.shape}, {self.img_size}"

        coeffs = sio.loadmat(pose_f)
        angles = torch.FloatTensor(coeffs["angle"])
        trans = torch.FloatTensor(coeffs["trans"])

        w2c_mat = compute_w2c_mat_from_estimated_pose_ffhq(
            angles, trans, self.sphere_center, sphere_r=self.sphere_r, normalize_trans=True
        )

        # # [B, 1]
        # pred_yaws, pred_pitches = compute_pitch_yaw_from_w2c_mat(w2c_mat, self.sphere_center_vec)
        # pred_yaws = pred_yaws[:, 0]
        # pred_pitches = pred_pitches[:, 0]

        # NOTE: Deep3DFace use angles for rotation in the order of Rx, Ry, Rz (Rx: rotation along x-axis).
        # - They define coordinate system as +X right, +Y up, +Z backward.
        # - We use +X right, +Y down, +Z up
        # Therefore:
        # - our -1 * yaw corresponds to their 2nd angle
        # - our pitch corresponds to their 1st angle
        # - our -1 roll corresponds to their 3rd angle. Though our roll is always zero.
        pred_yaws = -1 * torch.FloatTensor(angles[:, 1])
        pred_pitches = torch.FloatTensor(angles[:, 0])

        if self.flat_pose_dim == 9:
            # NOTE: we only condition on rotation
            flat_w2c_mat = w2c_mat[:, :3, :3].reshape((1, -1))[0, :]
        else:
            flat_w2c_mat = w2c_mat.reshape((1, -1))[0, :]

        return X, flat_w2c_mat, 0, pred_yaws, pred_pitches

    def get_file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()


class AFHQCat(Dataset):
    """AFHQCat Dataset"""

    def __init__(
        self,
        dataset_path,
        raw_img_size,
        img_size,
        pose_data_path,
        sphere_center,
        sphere_r=2.7,
        flat_pose_dim=9,
        **kwargs,
    ):
        super().__init__()

        self.dataset_path = dataset_path

        with open(os.path.join(pose_data_path, "dataset.json"), "r") as f:
            self.all_data = json.load(f)["labels"]

        print(f"\nFind {len(self.all_data)} valid images.\n")

        self.sphere_center = sphere_center
        self.sphere_center_vec = torch.FloatTensor([0, 0, sphere_center])
        self.sphere_r = sphere_r
        self.flat_pose_dim = flat_pose_dim

        assert len(self.all_data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.raw_img_size = raw_img_size
        self.img_size = img_size

        if self.raw_img_size != self.img_size:
            # transforms.InterpolationMode.LANCZOS
            transform_list = [transforms.Resize((img_size, img_size), interpolation=PIL.Image.LANCZOS)]
        else:
            transform_list = []

        # NOTE:
        # - ToTensor will change range from [0, 255] to [0, 1]
        # - The dataset has already been augmented by horizontal flip. We do not need to do it again.
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):

        img_fname, pose_info = self.all_data[index]

        img_f = os.path.join(self.dataset_path, img_fname)

        X = Image.open(img_f)

        assert np.array(X).shape[0] == self.raw_img_size, f"{np.array(X).shape}, {self.raw_img_size}"
        assert np.array(X).shape[1] == self.raw_img_size, f"{np.array(X).shape}, {self.raw_img_size}"

        X = self.transform(X)

        assert X.shape[1] == self.img_size, f"{X.shape}, {self.img_size}"
        assert X.shape[2] == self.img_size, f"{X.shape}, {self.img_size}"

        pnp_c2w_mats = np.array(pose_info[:16]).reshape((1, 4, 4))
        pnp_c2w_mats = torch.FloatTensor(pnp_c2w_mats)

        w2c_mat = compute_w2c_mat_from_estimated_pose_afhq(
            pnp_c2w_mats, self.sphere_center, sphere_r=self.sphere_r, normalize_trans=True
        )

        # [B, 1]
        pred_yaws, pred_pitches = compute_pitch_yaw_from_w2c_mat(w2c_mat, self.sphere_center_vec)

        pred_yaws = pred_yaws[:, 0]
        pred_pitches = pred_pitches[:, 0]

        if self.flat_pose_dim == 9:
            # NOTE: we only condition on rotation
            flat_w2c_mat = w2c_mat[:, :3, :3].reshape((1, -1))[0, :]
        else:
            flat_w2c_mat = w2c_mat.reshape((1, -1))[0, :]

        return X, flat_w2c_mat, 0, pred_yaws, pred_pitches


class MetFaces(Dataset):
    """MetFaces Dataset"""

    def __init__(
        self,
        dataset_path,
        raw_img_size,
        img_size,
        pose_data_path,
        sphere_center,
        sphere_r=1.0,
        flat_pose_dim=9,
        **kwargs,
    ):
        super().__init__()

        self.dataset_path = dataset_path

        if os.path.exists(os.path.join(pose_data_path, "fail_list.txt")):
            with open(os.path.join(pose_data_path, "fail_list.txt"), "r") as f:
                fail_list = [_.strip() for _ in f.readlines()]
        else:
            fail_list = []

        all_im_path = [os.path.join(dataset_path, i) for i in sorted(os.listdir(dataset_path)) if i.endswith("png")]
        # filter out failing cases
        im_path = []
        for elem in all_im_path:
            if os.path.basename(elem) not in fail_list:
                im_path.append(elem)
        print(f"\nFind {len(im_path)} valid images from {len(all_im_path)} images.\n")

        lm_path = [i.replace("png", "mat") for i in im_path]
        pose_data = [os.path.join(pose_data_path, "coeffs", os.path.basename(i)) for i in lm_path]

        self.data = list(zip(im_path, pose_data))

        print(f"\nFind {len(self.data)} valid images.\n")
        print("\n", self.data[:5], "\n")

        self.sphere_center = sphere_center
        self.sphere_center_vec = torch.FloatTensor([0, 0, sphere_center])
        self.sphere_r = sphere_r
        self.flat_pose_dim = flat_pose_dim

        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.raw_img_size = raw_img_size
        self.img_size = img_size

        if self.raw_img_size != self.img_size:
            # transforms.InterpolationMode.LANCZOS
            transform_list = [transforms.Resize((img_size, img_size), interpolation=PIL.Image.LANCZOS)]
        else:
            transform_list = []

        # NOTE:
        # - ToTensor will change range from [0, 255] to [0, 1]
        # - The dataset has already been augmented by horizontal flip. We do not need to do it again.
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_f, pose_f = self.data[index]

        with open(img_f, "rb") as fin:
            X = pyspng.load(fin.read())
        assert X.shape[0] == self.raw_img_size, f"{X.shape}, {self.raw_img_size}"
        assert X.shape[1] == self.raw_img_size, f"{X.shape}, {self.raw_img_size}"

        X = self.transform(Image.fromarray(X))

        assert X.shape[1] == self.img_size, f"{X.shape}, {self.img_size}"
        assert X.shape[2] == self.img_size, f"{X.shape}, {self.img_size}"

        coeffs = sio.loadmat(pose_f)
        angles = torch.FloatTensor(coeffs["angle"])
        trans = torch.FloatTensor(coeffs["trans"])

        w2c_mat = compute_w2c_mat_from_estimated_pose_ffhq(
            angles, trans, self.sphere_center, sphere_r=self.sphere_r, normalize_trans=True
        )

        # # [B, 1]
        # pred_yaws, pred_pitches = compute_pitch_yaw_from_w2c_mat(w2c_mat, self.sphere_center_vec)
        # pred_yaws = pred_yaws[:, 0]
        # pred_pitches = pred_pitches[:, 0]

        # NOTE: Deep3DFace use angles for rotation in the order of Rx, Ry, Rz (Rx: rotation along x-axis).
        # - They define coordinate system as +X right, +Y up, +Z backward.
        # - We use +X right, +Y down, +Z up
        # Therefore:
        # - our -1 * yaw corresponds to their 2nd angle
        # - our pitch corresponds to their 1st angle
        # - our -1 roll corresponds to their 3rd angle. Though our roll is always zero.
        pred_yaws = -1 * torch.FloatTensor(angles[:, 1])
        pred_pitches = torch.FloatTensor(angles[:, 0])

        if self.flat_pose_dim == 9:
            # NOTE: we only condition on rotation
            flat_w2c_mat = w2c_mat[:, :3, :3].reshape((1, -1))[0, :]
        else:
            flat_w2c_mat = w2c_mat.reshape((1, -1))[0, :]

        return X, flat_w2c_mat, 0, pred_yaws, pred_pitches


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8,
    )
    return dataloader, 3


def seed_worker(worker_id):
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_distributed(name, world_size, rank, batch_size, num_workres=4, torch_g=None, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workres,
        worker_init_fn=seed_worker,
        generator=torch_g,
    )

    return dataloader, sampler, 3
