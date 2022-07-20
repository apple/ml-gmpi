import os
import random
from typing import List, Optional

import imageio
import numpy as np
import torch
import tqdm


def create_output_dir(args) -> str:
    # If it has not ...
    # Create a new directory within the results folder with details of the
    # training parameters in the name
    output_dir = f"lr={args.lr}.bs={args.batch_size}"
    output_dir = os.path.join("output", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def set_reproduce(manualSeed: int = None):
    # Set random seed for reproducibility
    if manualSeed is None:
        manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    disable_tqdm=False,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        macro_block_size=1,
        **kwargs,
    )
    # print(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images, disable=disable_tqdm):
        writer.append_data(im)
    writer.close()
