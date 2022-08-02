<h1 align="center">Training and Evaluation</h1>

## Table of Contents

- [Training](#training)
  - [Set Up Virtual Environments](#set-up-virtual-environments)
  - [Download StyleGAN2 Checkpoints](#download-styleGAN2-checkpoints)
  - [Preprocess Data](#preprocess-data)
  - [Train](#train)
- [Evaluation](#evaluation)

# Training

Assume `GMPI_ROOT` represents the path to this repo:
```bash
cd /path/to/this/repo
export GMPI_ROOT=$PWD
```

## Set Up Virtual Environments

We need [MTCNN](https://github.com/ipazc/mtcnn), [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21), and [DeepFace](https://github.com/serengil/deepface) to complete the data processing and evaluation steps.

### MTCNN and DeepFace

We provide the `conda` environment yaml files for MTCNN and DeepFace:
- [mtcnn_env.yaml](../virtual_envs/mtcnn_env.yaml) for MTCNN;
- [deepface_env.yaml](../virtual_envs/deepface_env.yaml) for DeepFace.
```bash
conda env create -f mtcnn_env.yaml      # mtcnn_env
conda env create -f deepface_env.yaml   # deepface
```

### Deep3DFaceRecon_pytorch

**Note: we made small modifications to the original repo. Please use [our modified version](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch).** Please follow [the official instruction](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#requirements) to setup the virtual environments and to download the pretrained models. There are two major steps:
1. Install some packages and setup the environment: see [this link](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#installation);
2. Download some data: see [this link](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#prepare-prerequisite-models).

Assume the code repo locates at `Deep3DFaceRecon_PATH`:
```bash
export Deep3DFaceRecon_PATH=/path/to/Deep3DFaceRecon_pytorch
```

## Download StyleGAN2 Checkpoints

Download [StyleGAN2's pretrained checkpoints](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/):
```bash
mkdir -p ${GMPI_ROOT}/ckpts/stylegan2_pretrained/transfer-learning-source-nets/
cd ${GMPI_ROOT}/ckpts/stylegan2_pretrained
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl ./transfer-learning-source-nets    # FFHQ256
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl ./transfer-learning-source-nets   # FFHQ512
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl ./transfer-learning-source-nets  # FFHQ1024
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl .   # AFHQCat
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl .   # MetFaces
```

## Preprocess Data

We assume all data is placed under `${GMPI_ROOT}/runtime_dataset`.

We provide scripts in [data_preprocess](../data_preprocess) for steps described below.

### FFHQ

1. Please follow [StyleGAN2's guidance](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/README.md#preparing-datasets) to extract images from the raw TFRecords. After this step, you will obtain zip files with `ffhq256x256.zip` (~13G), `ffhq512x512.zip` (~52G), and `ffhq1024x1024.zip` (~206G). Place them under `${GMPI_ROOT}/runtime_dataset`.

2. We utilize [MTCNN](https://github.com/ipazc/mtcnn) to detect facial landmarks.

3. We use [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) to estimate poses for FFHQ. 

```bash
export RES=256

# landmark detection with MTCNN
conda activate mtcnn_env
python ${GMPI_ROOT}/data_preprocess/prepare_landmarks_ffhq.py --input_zipf ${GMPI_ROOT}/runtime_dataset/ffhq${RES}x${RES}.zip --save_dir ${GMPI_ROOT}/runtime_dataset/mtcnn_ffhq_${RES}

# run pose detection with Deep3DFaceRecon
conda activate deep3d_pytorch
cd ${Deep3DFaceRecon_PATH}
python estimate_pose_ffhq.py --name=pretrained --epoch=20 --img_folder=${GMPI_ROOT}/runtime_dataset/dummy --gmpi_img_res ${RES} --gmpi_root ${GMPI_ROOT}

# move pose results to GMPI
mv ${Deep3DFaceRecon_PATH}/checkpoints/pretrained/results/ffhq${RES}x${RES}/epoch_20_000000 ${GMPI_ROOT}/runtime_dataset/ffhq${RES}_deep3dface_coeffs
mv ${GMPI_ROOT}/runtime_dataset/mtcnn_ffhq_${RES}/detections/fail_list.txt ${GMPI_ROOT}/runtime_dataset/ffhq256_deep3dface_coeffs/
```

### AFHQCat

We use the same processed AFHQCat dataset from [EG3D](https://github.com/NVlabs/eg3d). We thank Eric Ryan Chan for providing the processed data. Please follow [the instructions](https://github.com/NVlabs/eg3d/blob/0b38adcc2bed6b4fda922efd6ec747e1216dc1fd/README.md#preparing-datasets) to obtain the AFHQCat dataset and rename the resulting folder to `afhq_v2_train_cat_512`.

### MetFaces

1. Please download the aligned-and-cropped version from [the official MetFaces website](https://github.com/NVlabs/metfaces-dataset).

2. We utilize [MTCNN](https://github.com/ipazc/mtcnn) to detect facial landmarks. Meanwhile, we augment the dataset by horizontally flipping the image.

3. We use [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) to estimate poses for MetFaces.

```bash
# we assume the raw images are stored in a folder with name metfaces1024x1024
mv ${GMPI_ROOT}/runtime_dataset/metfaces1024x1024 ${GMPI_ROOT}/runtime_dataset/metfaces1024x1024_xflip

conda activate mtcnn_env
# generate flipped dataset
python ${GMPI_ROOT}/data_preprocess/prepare_landmarks_metfaces.py --data_dir ${GMPI_ROOT}/runtime_dataset/metfaces1024x1024_xflip --save_dir ${GMPI_ROOT}/runtime_dataset/metfaces_detect --xflip 1
# detect landmarks wtih MTCNN
python ${GMPI_ROOT}/data_preprocess/prepare_landmarks_metfaces.py --data_dir ${GMPI_ROOT}/runtime_dataset/metfaces1024x1024_xflip --save_dir ${GMPI_ROOT}/runtime_dataset/metfaces_detect

# run pose detection with Deep3DFaceRecon
conda activate deep3d_pytorch_fork
cd ${Deep3DFaceRecon_PATH}
python estimate_pose_metfaces.py --name=pretrained --epoch=20 --img_folder=${GMPI_ROOT}/runtime_dataset/dummy --gmpi_root ${GMPI_ROOT}

# move data back to GMPI
mkdir -p ${GMPI_ROOT}/runtime_dataset/metfaces_xflip_deep3dface_coeffs
mv ${Deep3DFaceRecon_PATH}/checkpoints/pretrained/results/metfaces1024x1024_xflip/epoch_20_000000 ${GMPI_ROOT}/runtime_dataset/metfaces_xflip_deep3dface_coeffs/coeffs
mv ${GMPI_ROOT}/runtime_dataset/metfaces_detect/detections/fail_list.txt ${GMPI_ROOT}/runtime_dataset/metfaces_xflip_deep3dface_coeffs
```

### Processed Data

We provide processed poses for FFHQ and Metfaces in [this link](https://drive.google.com/drive/folders/1JDDtGXZP0Z8OwRROO5VNxOjQgwpUSJDj?usp=sharing).

### Final Folder Structure

If everyting goes well, you should observe the following folder structure:
```
.
+-- ckpts
|  +-- stylegan2_pretrained                # folder
|  |  +-- afhqcat.pkl                      # file
|  |  +-- metfaces.pkl                     # file
|  |  +-- transfer-learning-source-nets    # folder
+-- runtime_dataset
|  +-- ffhq256x256.zip                     # file
|  +-- ffhq256_deep3dface_coeffs           # folder
|  +-- ffhq512x512.zip                     # file
|  +-- ffhq512_deep3dface_coeffs           # folder
|  +-- ffhq1024x1024.zip                   # file
|  +-- ffhq1024_deep3dface_coeffs          # folder
|
|  +-- afhq_v2_train_cat_512               # folder
|
|  +-- metfaces1024x1024_xflip             # folder
|  +-- metfaces_xflip_deep3dface_coeffs    # folder
``` 

## Train

Run the following command to start training GMPI. Results will be saved in `${GMPI_ROOT}/experiments`. We use 8 Tesla V100 GPUs in our experiments. We recommend 32GB GPU memory if you want to train at a resolution of 1024x1024.

```bash
python launch.py \
--run_dataset FFHQ1024 \
--nproc_per_node 1 \
--task-type gmpi \
--run-type train \
--master_port 8370
```

- `run_dataset` can be in `["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"]`.
- Set `nproc_per_node` to be the number of GPUs you want to use.

### Generator Variants

This repo supports the following variants of the generator:
1. Vanilla version without alpha maps condition on depth or learnable token: set `torgba_cond_on_pos_enc: "none"` and `torgba_cond_on_pos_enc_embed_func: "none"` in [`configs/gmpi.yaml`](../configs/gmpi.yml);
2. Alpha maps conditions on normalized_depth: set `torgba_cond_on_pos_enc: "normalize_add_z"` and `torgba_cond_on_pos_enc_embed_func: "modulated_lrelu"` in [`configs/gmpi.yaml`](../configs/gmpi.yml);
3. Alpha maps comes from learnable tokens: set `torgba_cond_on_pos_enc: "normalize_add_z"` and `torgba_cond_on_pos_enc_embed_func: "learnable_param"` in [`configs/gmpi.yaml`](../configs/gmpi.yml);
4. Alpha maps come from predicted depth map: set `torgba_cond_on_pos_enc: "depth2alpha"` and `torgba_cond_on_pos_enc_embed_func: "modulated_lrelu"` in [`configs/gmpi.yaml`](../configs/gmpi.yml).

In the paper, we use the second variant: alpha maps conditions on normalized_depth.

# Evaluation

The command to evaluate the trained model is in [eval.sh](../gmpi/eval/eval.sh). We provide scripts to compute the following:
- FID/KID,
- Identity metrics,
- Depth metrics,
- Pose accuracy metrics.

Run the following command to evalute the model:
```bash
bash ${GMPI_ROOT}/gmpi/eval/eval.sh \
  ${GMPI_ROOT} \
  FFHQ512 \     # this can be FFHQ256, FFHQ512, FFHQ1024, AFHQCat, or MetFaces
  exp_id \      # this is your experiment ID
  ${Deep3DFaceRecon_PATH} \
  nodebug       # set this to "debug" to test your path for computing FID/KID is correct
```