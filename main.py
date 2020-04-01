import argparse
import cv2
import numpy as np
import os

import data
import utils
from distutils.util import strtobool


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--test_on", type=str, default="image")
    parser.add_argument("--show_results", type=str, default="True")
    parser.add_argument("--save_results_to", type=str, default="results")
    args_dict = parser.parse_args(args)
    return args_dict


def create_images(args):
    print("Loading images...")
    if args.test_on == "image":
        images, paths = data.images_from_paths([args.path])
    elif args.test_on == "paths_from_text_file":
        images, paths = data.images_from_paths(args.path)
    elif args.test_on == "cfd":
        or_im_paths, gt_paths = data.paths_generator_cfd(args.path)
        images, paths = data.images_from_paths(or_im_paths)
    elif args.test_on == "aigle-rn":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(args.path, "AIGLE_RN")
        images, paths = data.images_from_paths(or_im_paths)
    elif args.test_on == "esar":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(args.path, "ESAR")
        images, paths = data.images_from_paths(or_im_paths)
    print("Images loaded!")
    return images, paths


def main(args):
    if args.save_results_to is not None:
        if not os.path.exists(args.save_results_to):
            os.makedirs(args.save_results_to)
    images, paths = create_images(args)
    for idx, image in enumerate(images):
        cracks, preprocessing = utils.find_cracks(image)
        overlay = np.maximum((image / 2).astype(np.uint8), cracks)
        results = np.concatenate((image, preprocessing[1], preprocessing[2], preprocessing[3], overlay), axis=1)

        if strtobool(args.show_results):
            cv2.imshow("original / bh / filter_0 / filter_1 / overlay", results)
            cv2.waitKey(1000)

        if args.save_results_to is not None:
            image_name = os.path.split(paths[idx])[-1]
            cv2.imwrite(os.path.join(args.save_results_to, image_name), results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
