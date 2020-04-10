import data
import numpy as np
import os
import scipy.io


def create_images(dataset_name, dataset_path, mat_path=None):
    print("Loading images...")
    if dataset_name == "cfd":
        or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
    elif dataset_name == "aigle-rn":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
    elif dataset_name == "esar":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")
    ground_truth, gt_paths = data.images_from_paths(gt_paths)
    ground_truth = np.ascontiguousarray(ground_truth, dtype=np.float32)/255
    if mat_path is None:
        images, feature_names, file_names = get_morphological_features(or_im_paths, dataset_name)
    else:
        images, feature_names, file_names = open_morphological_features(mat_path)
    print("Images loaded!")
    return images, ground_truth, feature_names, file_names


def get_morphological_features(paths, dataset_name):
    paths = ";".join(paths)
    command = "matlab -nodesktop -nojvm -r 'try preprocess_images(\"%s\",\"%s\"); catch; end; quit'" % (
        paths, dataset_name)
    os.system(command)
    images, feature_names, file_names = open_morphological_features(dataset_name + ".mat")
    return images, feature_names, file_names


def open_morphological_features(path_to_mat):
    mat_root = os.path.split(path_to_mat)[0]

    mat_files = sorted([f for f in os.listdir(path_to_mat)
                        if not f.startswith(".") and f.endswith(".mat")],
                       key=lambda f: f.lower())
    images = np.ascontiguousarray(
        [scipy.io.loadmat(os.path.join(path_to_mat, mat_file))["images"] for mat_file in mat_files], dtype=np.float32)

    try:
        feature_names = scipy.io.loadmat(os.path.join(mat_root, "feature_names.mat"))["feature_names"]
        for feature in range(len(feature_names)):
            feature_names[feature] = feature_names[feature].strip()
    except FileNotFoundError:
        print("No mat file found for feature names.")
        feature_names = None

    return images, feature_names, mat_files


def flatten_pixels(images_array):
    print("Flattening images.")
    array_shape = images_array.shape

    if len(array_shape) == 3:
        return np.ascontiguousarray(
            np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2],), "F")), array_shape
    elif len(array_shape) == 4:
        return np.ascontiguousarray(
            np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2], array_shape[3]),
                       "F")), array_shape


def reconstruct_from_flat_pixels(flatten_pixels_array, original_shape):
    print("Reconstructing images.")
    flatten_shape = flatten_pixels_array.shape
    return np.reshape(flatten_pixels_array, original_shape, "F"), flatten_shape
