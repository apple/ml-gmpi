import argparse
import os
import zipfile

import numpy as np
import PIL
import tqdm

try:
    import pyspng
except ImportError:
    pyspng = None

from mtcnn import MTCNN

N_IMGS = 1


def get_file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def main(args):

    input_zipf = args.input_zipf

    detect_base_dir = os.path.join(args.save_dir, "detections")
    detect_res_dir = os.path.join(detect_base_dir, "results")
    os.makedirs(detect_res_dir, exist_ok=True)

    detector = MTCNN()

    # NOTE: we always use this zipfile object to be fast: https://stackoverflow.com/a/37148834
    zip_obj = zipfile.ZipFile(input_zipf)

    all_f_list = zip_obj.namelist()

    PIL.Image.init()
    all_img_f_list = [_ for _ in all_f_list if get_file_ext(_) in PIL.Image.EXTENSION]
    sorted_f_list = sorted(all_img_f_list)
    print("\nsorted_f_list: ", len(sorted_f_list), sorted_f_list[:5], "\n")

    for i, filename in tqdm.tqdm(enumerate(sorted_f_list), total=len(sorted_f_list)):

        sub_folder = os.path.join(detect_res_dir, filename.split("/")[0])
        os.makedirs(sub_folder, exist_ok=True)

        basename = os.path.splitext(filename)[0]

        with zip_obj.open(filename, "r") as f:
            if pyspng is not None and get_file_ext(filename) == ".png":
                img = pyspng.load(f.read())
            else:
                img = np.array(PIL.Image.open(f))

        text_path = f"{detect_res_dir}/{basename}.txt"
        result = detector.detect_faces(img)
        try:
            keypoints = result[0]["keypoints"]
            with open(text_path, "w") as f:
                for value in keypoints.values():
                    f.write(f"{value[0]}\t{value[1]}\n")
                # print(f"File successfully written: {text_path}")
        except:
            if i == 0:
                mode = "w"
            else:
                mode = "a"
            with open(os.path.join(detect_base_dir, "fail_list.txt"), mode) as fail_f:
                fail_f.write(f"{filename}\n")
            print("\n", filename, filename, "\n")

    if not os.path.exists(os.path.join(detect_base_dir, "fail_list.txt")):
        with open(os.path.join(detect_base_dir, "fail_list.txt"), "w") as fail_f:
            fail_f.write(f"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get landmarks from images.")
    parser.add_argument("--input_zipf", type=str, default=None, help="zip file with the input images")
    parser.add_argument("--save_dir", type=str, default=None, help="zip file with the input images")
    args = parser.parse_args()

    main(args)
