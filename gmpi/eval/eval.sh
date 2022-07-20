#!/bin/bash
{

REPO_DIR="$1" 
DATASET="$2"
EXP_ID="$3"
Deep3DFaceRecon_PATH="$4"
DEBUG="$5"

export PYTHONPATH=${REPO_DIR}/gmpi/models:${REPO_DIR}:$PYTHONPATH

SEED=123

if [ "${DEBUG}" == "debug" ]; then
    N_IMGS=2048
else
    N_IMGS=50000
fi

REAL_BASE_DIR="${REPO_DIR}/runtime_dataset/real_data"
FAKE_BASE_DIR="${REPO_DIR}/ckpts/gmpi_pretrained"


if [ "${DATASET}" == "FFHQ256" ]; then
    RES=256
    ABB=FFHQ
elif [ "${DATASET}" == "FFHQ512" ]; then
    RES=512
    ABB=FFHQ
elif [ "${DATASET}" == "FFHQ1024" ]; then
    RES=1024
    ABB=FFHQ
elif [ "${DATASET}" == "MetFaces" ]; then
    RES=1024
    ABB=FFHQ
elif [ "${DATASET}" == "AFHQCat" ]; then
    RES=512
    ABB=AFHQCat
else
    RES=0
fi

if [ "${EXP_ID}" == "stylegan2_sanity_check" ]; then
    SANITY_CHECK=1
else
    SANITY_CHECK=0
fi

REAL_DIR=${REAL_BASE_DIR}/${DATASET}/${ABB}_real_res_${RES}_n_${N_IMGS}

N_PLANES=96

TRUNCATION_PSI=1.0

CKPT_DIR=${FAKE_BASE_DIR}/${EXP_ID}
FAKE_DIR=${CKPT_DIR}/planes_${N_PLANES}_n_${N_IMGS}/psi_${TRUNCATION_PSI}

N_CONSIST=1024
CONSIST_DIR=${CKPT_DIR}/planes_${N_PLANES}_n_${N_CONSIST}/psi_${TRUNCATION_PSI}

N_GEO=1024
GEO_DIR=${CKPT_DIR}/planes_${N_PLANES}_n_${N_CONSIST}/psi_${TRUNCATION_PSI}

# --------------------------------------------------------------------------------------------
# For FID/KID

# save real images to disk
eval "$(conda shell.bash hook)"
conda activate gmpi
export MKL_NUM_THREADS=5 && export NUMEXPR_NUM_THREADS=5 && \
python ${REPO_DIR}/gmpi/eval/prepare_real_data.py \
--dataset ${DATASET} \
--seed ${SEED} \
--save_dir ${REAL_BASE_DIR} \
--exp_config ${REPO_DIR}/configs/gmpi.yml \
--n_imgs ${N_IMGS}

eval "$(conda shell.bash hook)"
conda activate gmpi
export MKL_NUM_THREADS=5 && export NUMEXPR_NUM_THREADS=5 && \
python ${REPO_DIR}/gmpi/eval/prepare_fake_data.py \
--ckpt_path ${CKPT_DIR}/generator.pth \
--seed ${SEED} \
--save_dir ${FAKE_DIR} \
--nplanes ${N_PLANES} \
--n_imgs ${N_IMGS} \
--task fid_kid \
--truncation_psi ${TRUNCATION_PSI} \
--exp_config ${CKPT_DIR}/config.pth \
--dataset ${DATASET} \
--stylegan2_sanity_check ${SANITY_CHECK}

# FID and KID
eval "$(conda shell.bash hook)"
conda activate gmpi
python ${REPO_DIR}/gmpi/eval/compute_fid_kid.py \
--real_dir ${REAL_DIR} \
--fake_dir ${FAKE_DIR}/fid_kid/rgb \
--save_dir ${FAKE_DIR}/fid_kid


# --------------------------------------------------------------------------------------------
# For identity metrices

eval "$(conda shell.bash hook)"
conda activate gmpi
export MKL_NUM_THREADS=5 && export NUMEXPR_NUM_THREADS=5 && \
python ${REPO_DIR}/gmpi/eval/prepare_fake_data.py \
--ckpt_path ${CKPT_DIR}/generator.pth \
--seed ${SEED} \
--save_dir ${CONSIST_DIR} \
--nplanes ${N_PLANES} \
--n_imgs ${N_CONSIST} \
--task consistency \
--exp_config ${CKPT_DIR}/config.pth \
--truncation_psi ${TRUNCATION_PSI} \
--dataset ${DATASET} \
--stylegan2_sanity_check ${SANITY_CHECK}

eval "$(conda shell.bash hook)"
conda activate deepface
python ${REPO_DIR}/gmpi/eval/compute_consistency.py \
--input_dir ${CONSIST_DIR}/consistency \
--n_imgs ${N_CONSIST} \
--nproc 2


# --------------------------------------------------------------------------------------------
# for Depth and pose

eval "$(conda shell.bash hook)"
conda activate gmpi
export MKL_NUM_THREADS=5 && export NUMEXPR_NUM_THREADS=5 && \
python ${REPO_DIR}/gmpi/eval/prepare_fake_data.py \
--ckpt_path ${CKPT_DIR}/generator.pth \
--seed ${SEED} \
--save_dir ${GEO_DIR} \
--nplanes ${N_PLANES} \
--n_imgs ${N_GEO} \
--task geometry \
--exp_config ${CKPT_DIR}/config.pth \
--save_depth 1 \
--dataset ${DATASET} \
--truncation_psi ${TRUNCATION_PSI} \
--stylegan2_sanity_check ${SANITY_CHECK}

# prepare landmarks
# we use MTCNN (https://github.com/ipazc/mtcnn)
eval "$(conda shell.bash hook)"
conda activate mtcnn_env
export MKL_NUM_THREADS=5 && export NUMEXPR_NUM_THREADS=5 &&
python ${REPO_DIR}/gmpi/eval/prepare_face_landmarks.py \
--data_dir ${GEO_DIR}/geometry/rgb

# require Deep3DFace (https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch)
eval "$(conda shell.bash hook)"
conda activate deep3d_pytorch
cd ${Deep3DFaceRecon_PATH}
python ${Deep3DFaceRecon_PATH}/estimate_pose_gmpi.py \
--name=pretrained \
--epoch=20 \
--gmpi_img_root ${GEO_DIR}/geometry/rgb \
--gmpi_depth_root ${GEO_DIR}/geometry/depth \
--gmpi_detect_root ${GEO_DIR}/geometry/rgb/detections

# compute error
eval "$(conda shell.bash hook)"
conda activate gmpi
python ${REPO_DIR}/gmpi/eval/compute_geometry.py \
--geo_dir ${GEO_DIR}/geometry \
--angle_err 1

exit;
}