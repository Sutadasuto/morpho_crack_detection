import cv2
import numpy as np
import os


def paths_generator_crack_dataset(training_data_path):
    images_path, dataset = os.path.split(training_data_path)
    if dataset == "ESAR":
        file_end = ".jpg"
    elif dataset == "AIGLE_RN":
        file_end = "or.png"
    ground_truth_path = os.path.join(os.path.split(images_path)[0], "GROUND_TRUTH", dataset)

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(training_data_path, "Im_" + os.path.split(f)[-1].replace(".png", file_end)) for
                            f in ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_cfd(training_data_path):
    root_path, dataset = os.path.split(training_data_path)
    ground_truth_path = os.path.join(root_path, "groundTruthPng")

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and f.endswith(".png")],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(training_data_path, os.path.split(f)[-1].replace(".png", ".jpg")) for f in
                            ground_truth_image_paths]
    # training_image_paths = sorted(
    #     [os.path.join(training_data_path, f) for f in os.listdir(training_data_path)[:len(ground_truth_image_paths)]
    #      if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))],
    #     key=lambda f: f.lower())

    return training_image_paths, ground_truth_image_paths


def images_from_paths(paths):
    if type(paths) is str:
        with open(paths, "r") as f:
            paths = f.readlines()
    data = []
    for idx, path in enumerate(paths):
        img = cv2.imread(path.strip(), cv2.IMREAD_GRAYSCALE)
        data.append(img)
    return data

