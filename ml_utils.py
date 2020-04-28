import cv2
import data
import numpy as np
import os
import scipy.io

from sklearn.metrics import accuracy_score, r2_score


# def create_images(dataset_name, dataset_path, mat_path=None):
#     print("Loading images...")
#     if dataset_name == "cfd":
#         or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
#     elif dataset_name == "aigle-rn":
#         or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
#     elif dataset_name == "esar":
#         or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")
#     ground_truth, gt_paths = data.images_from_paths(gt_paths)
#     ground_truth = np.array(ground_truth, dtype=np.float32) / 255
#     if mat_path is None:
#         images, feature_names, file_names = get_morphological_features(or_im_paths, dataset_name)
#     else:
#         images, feature_names, file_names = open_morphological_features(mat_path)
#     print("Images loaded!")
#     return images, ground_truth, feature_names, file_names
#
#
# def get_morphological_features(paths, dataset_name):
#     paths = ";".join(paths)
#     command = "matlab -nodesktop -nojvm -r 'try preprocess_images(\"%s\",\"%s\"); catch; end; quit'" % (
#         paths, dataset_name)
#     os.system(command)
#     images, feature_names, file_names, or_im_shape = open_morphological_features(dataset_name + ".mat")
#     return images, feature_names, file_names
#
#
# def open_morphological_features(path_to_mat):
#     mat_root = os.path.split(path_to_mat)[0]
#
#     mat_files = sorted([f for f in os.listdir(path_to_mat)
#                         if not f.startswith(".") and f.endswith(".mat")],
#                        key=lambda f: f.lower())
#     images = np.array([scipy.io.loadmat(os.path.join(path_to_mat, mat_file))["images"] for mat_file in mat_files],
#                       dtype=np.float32)
#
#     try:
#         feature_names = scipy.io.loadmat(os.path.join(mat_root, "feature_names.mat"))["feature_names"]
#         for feature in range(len(feature_names)):
#             feature_names[feature] = feature_names[feature].strip()
#     except FileNotFoundError:
#         print("No mat file found for feature names.")
#         feature_names = None
#
#     return images, feature_names, mat_files


def create_samples(dataset_name, dataset_path, mat_path=None, balanced=False, save_images=True):
    print("Loading data...")

    if dataset_name == "cfd" or dataset_name == "cfd-pruned":
        or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
    elif dataset_name == "aigle-rn":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
    elif dataset_name == "esar":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")

    if mat_path is not None:
        features, labels, feature_names, selected_pixels = open_morphological_features(mat_path, balanced)
        print("Data loaded!")
        return features, labels, feature_names, selected_pixels, [or_im_paths, gt_paths]

    gt_intensity = 255 if dataset_name == "cfd" or dataset_name == "cfd-pruned" else 0
    features, labels, feature_names, selected_pixels = get_morphological_features(or_im_paths, gt_paths, gt_intensity,
                                                                          dataset_name, balanced, save_images)
    print("Data loaded!")
    return features, labels, feature_names, selected_pixels, [or_im_paths, gt_paths]


def create_multidataset_samples(dataset_names, dataset_paths, mat_paths=[None, None], balanceds=[False, False], save_imagess=[True, True]):
    print("Loading data...")

    features_list, labels_list, feature_names_list, selected_pixels_list, paths_list = [],[],[],[],[]
    for idx in range(len(dataset_names)):
        dataset_name, dataset_path, mat_path, balanced, save_images = dataset_names[idx], dataset_paths[idx], mat_paths[idx], balanceds[idx], save_imagess[idx]

        if dataset_name == "cfd" or dataset_name == "cfd-pruned":
            or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
        elif dataset_name == "aigle-rn":
            or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
        elif dataset_name == "esar":
            or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")

        if mat_path is not None:
            features, labels, feature_names, selected_pixels = open_morphological_features(mat_path, balanced)
            features_list.append(features)
            labels_list.append(labels)
            feature_names_list.append(feature_names)
            selected_pixels_list.append(selected_pixels)
            paths_list.append([or_im_paths, gt_paths])
            continue

        gt_intensity = 255 if dataset_name == "cfd" or dataset_name == "cfd-pruned" else 0
        features, labels, feature_names, selected_pixels = get_morphological_features(or_im_paths, gt_paths, gt_intensity,
                                                                                      dataset_name, balanced, save_images)
        features_list.append(features)
        labels_list.append(labels)
        feature_names_list.append(feature_names)
        selected_pixels_list.append(selected_pixels)
        paths_list.append([or_im_paths, gt_paths])

    print("Data loaded!")
    return features_list, labels_list, feature_names_list, selected_pixels_list, paths_list


def get_morphological_features(paths, gt_paths, gt_value, dataset_name, balanced, save_resulting_images):
    paths = ";".join(paths)
    gt_paths = ";".join(gt_paths)
    command = "matlab -nodesktop -nojvm -r 'try preprocess_images(\"%s\",\"%s\",%s,\"%s\",%s,%s); catch; end; quit'" % (
        paths, gt_paths, gt_value, dataset_name, str(balanced).lower(), str(save_resulting_images).lower())
    os.system(command)
    balanced_string = "_balanced" if balanced else ""
    features, labels, feature_names, selected_pixels = open_morphological_features(dataset_name + balanced_string + ".mat", balanced)
    return features, labels, feature_names, selected_pixels


def open_morphological_features(path_to_mat, balanced=False):
    mat_root, dataset_name = os.path.split(path_to_mat)

    if balanced is True:
        balanced_string = "_balanced"
    elif balanced > 0:
        balanced_string = "_1_to_%s" % str(balanced).replace(".", ",")
    elif balanced < 0:
        balanced_string = "_1_to_%s_weighted" % str(-balanced).replace(".", ",")
    else:
        balanced_string = ""

    dataset_name = dataset_name.split(balanced_string + ".mat")[0]
    features = scipy.io.loadmat(path_to_mat)["data"]
    labels = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_labels.mat"))["labels"]

    try:
        feature_names = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_feature_names.mat"))["feature_names"]
        for feature in range(len(feature_names)):
            feature_names[feature] = feature_names[feature].strip()
    except FileNotFoundError:
        print("No mat file found for feature names.")
        feature_names = None

    try:
        selected_pixels = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_pick_maps.mat"))["pick_maps"]
    except FileNotFoundError:
        print("No mat file found for picked pixels.")
        selected_pixels = None

    return features, np.ravel(labels), feature_names, selected_pixels


def flatten_pixels(images_array):
    print("Flattening images.")
    array_shape = images_array.shape

    if len(array_shape) == 3:
        return np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2],), "F"), array_shape
    elif len(array_shape) == 4:
        return np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2], array_shape[3]),
                          "F"), array_shape


def reconstruct_from_flat_pixels(flatten_pixels_array, original_shape):
    print("Reconstructing images.")
    flatten_shape = flatten_pixels_array.shape
    return np.reshape(flatten_pixels_array, original_shape, "F"), flatten_shape


def reconstruct_from_selected_pixels(selected_pixels, predicted_labels, real_labels, paths):
    or_paths = paths[0]
    gt_paths = paths[1]
    current_image = 0
    gt = cv2.imread(gt_paths[0])
    img = cv2.imread(or_paths[0])
    predicted_image = np.zeros(gt.shape, dtype=np.uint8)
    resulting_images = []

    for pixel in range(len(selected_pixels)):
        image, row, col = selected_pixels[pixel]
        if image > current_image:
            resulting_images.append(np.concatenate((img, gt, predicted_image), axis=1))
            gt = cv2.imread(gt_paths[image])
            img = cv2.imread(or_paths[image])
            predicted_image = np.zeros(gt.shape, dtype=np.uint8)
            current_image = image
        if predicted_labels[pixel] == 1 or predicted_labels[pixel] == 0:
            if predicted_labels[pixel] == real_labels[pixel]:
                if real_labels[pixel] == 1:
                    predicted_image[row, col, :] = np.array([0, 255, 0], dtype=np.uint8)
                else:
                    predicted_image[row, col, :] = np.array([255, 0, 0], dtype=np.uint8)
            else:
                predicted_image[row, col, :] = np.array([0, 0, 255], dtype=np.uint8)
        else:
            regression_value = max(0, predicted_labels[pixel])
            regression_value = min(1, regression_value)
            regression_value = 255*regression_value
            predicted_image[row, col, :] = np.array([regression_value, regression_value, regression_value], dtype=np.uint8)
    resulting_images.append(np.concatenate((img, gt, predicted_image), axis=1))
    return resulting_images


def save_visual_results(selected_pixels, predicted_labels, real_labels, paths, path_dir="resulting_images"):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    results = reconstruct_from_selected_pixels(selected_pixels, predicted_labels, real_labels, paths)
    for idx, image_path in enumerate(paths[1]):
        file_name = os.path.split(image_path)[1]
        cv2.imwrite(os.path.join(path_dir, file_name), results[idx])


def cross_validate_predict(data, folds, cv_results):

    n_samples, n_features = data.shape
    predicts = np.zeros((n_samples,))
    for idx, fold in enumerate(folds):
        indices = fold[1]
        estimator = cv_results["estimator"][idx]
        fold_results = estimator.predict(data[indices])
        for fold_idx, result in enumerate(fold_results):
            predicts[indices[fold_idx]] = result
    return predicts


def cross_dataset_validation(model, x_train, y_train, x_test, y_test, test_selected_pixels, test_paths, save_images_to=None, score_function=None):

    model.fit(x_train, y_train)
    cross_dataset_predictions = model.predict(x_test)
    if score_function is None:
        if set(cross_dataset_predictions) == {0, 1}:
            score_function = accuracy_score
        else:
            score_function = r2_score
    cross_dataset_score = score_function(y_test, cross_dataset_predictions)

    if save_images_to is not None:
        save_visual_results(test_selected_pixels, cross_dataset_predictions, y_test, test_paths, save_images_to)
    return cross_dataset_score, score_function.__name__
