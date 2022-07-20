import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback

import joblib
import numpy as np

# test with tensorflow-gpu==2.8.0
import tensorflow as tf
import tqdm
from deepface import DeepFace

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compute_metrics_subproc(subproc_input):

    (worker_id, idx_list, input_dir) = subproc_input

    all_res = {}
    final_res = []

    for i in tqdm.tqdm(idx_list):
        tmp_f1 = os.path.join(input_dir, f"rgb/{i:06d}_0.png")
        tmp_f2 = os.path.join(input_dir, f"rgb/{i:06d}_1.png")
        assert os.path.exists(tmp_f1), f"{tmp_f1}"
        assert os.path.exists(tmp_f2), f"{tmp_f2}"

        try:
            # Example:
            # {'verified': False, 'distance': 0.964447578670103, 'threshold': 0.68, 'model': 'ArcFace', 'detector_backend': 'opencv', 'similarity_metric': 'cosine'}
            tmp_obj = DeepFace.verify(
                tmp_f1,
                tmp_f2,
                model_name="ArcFace",
                distance_metric="cosine",
                enforce_detection=False,
                detector_backend="mtcnn",
            )
        except:
            traceback.print_exc()
            err = sys.exc_info()[0]
            print(err)
            print("\n", tmp_f1, tmp_f2, "\n")
            sys.exit(1)

        all_res[i] = tmp_obj
        final_res.append(1 - tmp_obj["distance"])

    return all_res, final_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_imgs", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    opt = parser.parse_args()

    print("\n[Consistency]: ", opt.input_dir, "\n")

    all_res = {}
    final_res = []

    idx_list = [[] for _ in range(opt.nproc)]
    for i, view_id in enumerate(range(opt.n_imgs)):
        idx_list[i % opt.nproc].append(view_id)

    # NOTE: np.matmul may freeze when using default "fork"
    # https://github.com/ModelOriented/DALEX/issues/412
    with mp.get_context("spawn").Pool(opt.nproc) as pool:
        gather_output = pool.map(
            compute_metrics_subproc,
            zip(
                range(opt.nproc),
                idx_list,
                [opt.input_dir for _ in range(opt.nproc)],
            ),
        )
        pool.close()
        pool.join()

    for elem in gather_output:
        all_res.update(elem[0])
        final_res.extend(elem[1])

    with open(os.path.join(opt.input_dir, "all_res.pt"), "wb") as f:
        joblib.dump(all_res, f, compress="lz4")

    with open(os.path.join(opt.input_dir, "all_res.json"), "w") as f:
        json.dump(all_res, f)

    aggregated_mean = np.mean(final_res)
    aggregated_std = np.std(final_res)

    with open(os.path.join(opt.input_dir, "aggregated.txt"), "w") as f:
        f.write(f"{aggregated_mean}, {aggregated_std}\n")

    print(aggregated_mean, aggregated_std)
