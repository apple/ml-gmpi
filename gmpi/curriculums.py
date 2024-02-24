# Modified from https://github.com/marcoamonteiro/pi-GAN

"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""


# fmt: off

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 1024) >= current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            return_dict.update(curriculum[curriculum_step])
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


FFHQ = {

    "res_dict": {
        256: {'batch_size': 8, 'num_steps': 32, 'img_size': 256, 'tex_size': 256, 'batch_split': 1, 'gen_lr': 0.002, 'disc_lr': 0.002},
        512: {'batch_size': 4, 'num_steps': 32, 'img_size': 512, 'tex_size': 512, 'batch_split': 1, 'gen_lr': 0.002, 'disc_lr': 0.002},
        1024: {'batch_size': 4, 'num_steps': 32, 'img_size': 1024, 'tex_size': 1024, 'batch_split': 2, 'gen_lr': 0.002, 'disc_lr': 0.002},
    },

    "res_dict_learnable_param": {
        256: {'batch_size': 4, 'num_steps': 32, 'img_size': 256, 'tex_size': 256, 'batch_split': 1, 'gen_lr': 0.002, 'disc_lr': 0.002},
        512: {'batch_size': 4, 'num_steps': 32, 'img_size': 512, 'tex_size': 512, 'batch_split': 2, 'gen_lr': 0.002, 'disc_lr': 0.002},
        1024: {'batch_size': 4, 'num_steps': 32, 'img_size': 1024, 'tex_size': 1024, 'batch_split': 2, 'gen_lr': 0.002, 'disc_lr': 0.002},
    },

    # D uses lr as StyleGAN2
    int(0): {'batch_size': 4, 'num_steps': 32, 'img_size': 1024, 'tex_size': 1024, 'batch_split': 1, 'gen_lr': 0.002, 'disc_lr': 0.002},
    int(200e3): {},

    'dataset_path': './FFHQ',
    "pose_data_path": "none",

    'fov': 12.6,
    'ray_start': 0.95,
    'ray_end': 1.12,

    'h_stddev': 0.289,
    'v_stddev': 0.127,
    'h_mean': 0.0,  # math.pi*0.5,  # NOTE: we have changed "sample_camera_positions" in order to use mean of 0
    'v_mean': 0.0,  # math.pi*0.5,

    'latent_dim': 512,
    'stylegan2_w_dim': 512,
    'generator_label_dim': 0,

    'fade_steps': 10000,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 10.0,
    'grad_clip': 10,
    'dataset': 'FFHQ',
    'z_dist': 'gaussian',
    "raw_img_size": 1024,
    "eval_img_size": 1024,
}


AFHQCat = {

    "res_dict": {
        512: {'batch_size': 4, 'num_steps': 32, 'img_size': 512, 'tex_size': 512, 'batch_split': 1, 'gen_lr': 0.002, 'disc_lr': 0.002},
    },

    # D uses lr as StyleGAN2
    int(0): {'batch_size': 1, 'num_steps': 32, 'img_size': 512, 'tex_size': 512, 'batch_split': 1, 'gen_lr': 0.0025, 'disc_lr': 0.0025},
    int(200e3): {},

    'dataset_path': './AFHQ',
    "pose_data_path": "none",
    'fov': 13.39,
    'ray_start': 2.55,
    'ray_end': 2.8,

    'h_stddev': 0.19,
    'v_stddev': 0.15,
    'h_mean': 0.0,  # math.pi*0.5,  # NOTE: we have changed "sample_camera_positions" in order to use mean of 0
    'v_mean': 0.0,  # math.pi*0.5,

    'latent_dim': 512,
    'stylegan2_w_dim': 512,
    'generator_label_dim': 0,

    'fade_steps': 10000,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 10.0,
    'grad_clip': 10,
    'dataset': 'AFHQCat',
    'z_dist': 'gaussian',
    "raw_img_size": 512,
    "eval_img_size": 512,
}


MetFaces = {
    
    "res_dict": {
        1024: {'batch_size': 4, 'num_steps': 32, 'img_size': 1024, 'tex_size': 1024, 'batch_split': 2, 'gen_lr': 0.002, 'disc_lr': 0.002},
    },

    # D uses lr as StyleGAN2
    int(0): {'batch_size': 4, 'num_steps': 32, 'img_size': 1024, 'tex_size': 1024, 'batch_split': 2, 'gen_lr': 0.002, 'disc_lr': 0.002},
    int(200e3): {},

    'dataset_path': './MetFaces',
    "pose_data_path": "none",
    'fov': 12.6,
    'ray_start': 0.95,
    'ray_end': 1.12,

    'h_stddev': 0.339,
    'v_stddev': 0.133,
    'h_mean': 0.0,  # math.pi*0.5,  # NOTE: we have changed "sample_camera_positions" in order to use mean of 0
    'v_mean': 0.0,  # math.pi*0.5,

    'latent_dim': 512,
    'stylegan2_w_dim': 512,
    'generator_label_dim': 0,

    'fade_steps': 10000,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 10.0,
    'grad_clip': 10,
    'dataset': 'MetFaces',
    'z_dist': 'gaussian',
    "raw_img_size": 1024,
    "eval_img_size": 1024,
}

# fmt: on
