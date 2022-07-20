import argparse
import json
import os

import joblib
import torch
from torch_fidelity import calculate_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nCompute FID/KID from \nreal: {opt.real_dir}\nfake: {opt.fake_dir}\n")

    metrics_dict = calculate_metrics(
        input1=opt.fake_dir,
        input2=opt.real_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=True,
        verbose=True,
    )
    print(metrics_dict)

    with open(os.path.join(opt.save_dir, "fid_kid.pt"), "wb") as f:
        joblib.dump(metrics_dict, f, compress="lz4")

    new_metrics_dict = {}
    for k in metrics_dict:
        new_metrics_dict[k] = str(metrics_dict[k])
    with open(os.path.join(opt.save_dir, "fid_kid.json"), "w") as f:
        json.dump(new_metrics_dict, f)
