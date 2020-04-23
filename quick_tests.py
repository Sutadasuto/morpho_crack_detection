import argparse
import cv2
import numpy as np
import os

import data
import morpho_utils
from distutils.util import strtobool
from skimage.filters import frangi


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
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    se_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for idx, image in enumerate(images):
        img = morpho_utils.min_max_contrast_enhancement(image)
        bottom_hat = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, se_2), cv2.MORPH_CLOSE, se)
        bottom_hat = np.maximum(np.zeros(img.shape, dtype=np.int16), bottom_hat.astype(np.int16) - img.astype(np.int16)).astype(np.uint8)
        directional_filtered = morpho_utils.multi_dir_linear_filtering(bottom_hat, 11, 15)
        # vesselness = frangi(1 - binary.astype(np.float32) / binary.max())
        # test_raro = img.astype(np.float32)/img.max() * vesselness.astype(np.float32)/vesselness.max()
        # blackhat = (255*cv2.morphologyEx(test_raro, cv2.MORPH_BLACKHAT, se)).astype(np.uint8)
        # blackhat_enhanced = np.maximum(np.zeros(img.shape, dtype=np.int16), blackhat.astype(np.int16) - cv2.medianBlur(blackhat, 11).astype(np.int16)).astype(np.uint8)
        # blackhat_connected = morpho_utils.multi_dir_linear_filtering(blackhat_enhanced, 11, 15)
        # blackhat_vessel = frangi(1 - blackhat_connected.astype(np.float32) / blackhat_connected.max())
        blackhat_vessel = bottom_hat.astype(np.float32) * (directional_filtered.astype(np.float32) / directional_filtered.max())
        blackhat_vessel = cv2.adaptiveThreshold(directional_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 51, 0)
        blackhat_vessel = cv2.morphologyEx(blackhat_vessel, cv2.MORPH_OPEN, se_2)
        blackhat_vessel = morpho_utils.filter_by_shape(blackhat_vessel)
        blackhat_vessel = cv2.medianBlur(directional_filtered, 3)

        if strtobool(args.show_results):
            cv2.imshow("original / bottom_hat / filtered / vesselness", np.concatenate((image.astype(np.float32)/image.max(), bottom_hat.astype(np.float32)/bottom_hat.max(), directional_filtered.astype(np.float32)/directional_filtered.max(), blackhat_vessel.astype(np.float32)/blackhat_vessel.max()), axis=1))
            cv2.waitKey(1000)



if __name__ == "__main__":
    args = parse_args()
    main(args)