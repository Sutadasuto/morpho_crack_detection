import cv2
import ml_utils
import numpy as np
import os

from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, KFold
from sklearn.svm import LinearSVC, SVC


def train_sgd_models():
    x, y, feature_names, file_names = ml_utils.create_images("cfd",
                                                             "/media/winbuntu/databases/CrackForestDataset")
    x, or_x_shape = ml_utils.flatten_pixels(x)
    y, or_y_shape = ml_utils.flatten_pixels(y)

    lasso = linear_model.Lasso()
    print("Cross-validating with " + str(type(lasso)))
    y_pred = cross_val_predict(lasso, x, y, cv=2, verbose=50)
    del x

    print("Preparing to show results.")
    y, _ = ml_utils.reconstruct_from_flat_pixels(y, or_y_shape)
    y_pred, _ = ml_utils.reconstruct_from_flat_pixels(y_pred, or_y_shape)

    for i in range(or_y_shape[0]):
        current_result = np.concatenate((y[i, :, :], y_pred[i, :, :]), axis=1) * 255
        current_result = np.maximum(current_result, np.zeros(current_result.shape))
        current_result = np.minimum(current_result, 255 * np.ones(current_result.shape))
        current_result = current_result.astype(np.uint8)
        if not os.path.exists("results ml"):
            os.makedirs("results ml")
        cv2.imwrite(os.path.join("results ml", file_names[i].replace(".mat", ".png")), current_result)
        # cv2.waitKey(1000)


# x, y, feature_names, file_names = ml_utils.create_images("cfd",
#                                                          "/media/winbuntu/databases/CrackForestDataset")
# x, or_x_shape = ml_utils.flatten_pixels(x)
# y, or_y_shape = ml_utils.flatten_pixels(y)
#
# lasso = linear_model.Lasso()
# print("Cross-validating with " + str(type(lasso)))
# y_pred = cross_val_predict(lasso, x, y, cv=2, verbose=50)
# del x
#
# print("Preparing to show results.")
# y, _ = ml_utils.reconstruct_from_flat_pixels(y, or_y_shape)
# y_pred, _ = ml_utils.reconstruct_from_flat_pixels(y_pred, or_y_shape)
#
# for i in range(or_y_shape[0]):
#     current_result = np.concatenate((y[i, :, :], y_pred[i, :, :]), axis=1) * 255
#     current_result = np.maximum(current_result, np.zeros(current_result.shape))
#     current_result = np.minimum(current_result, 255*np.ones(current_result.shape))
#     current_result = current_result.astype(np.uint8)
#     if not os.path.exists("results ml"):
#         os.makedirs("results ml")
#     cv2.imwrite(os.path.join("results ml", file_names[i].replace(".mat", ".png")), current_result)
#     # cv2.waitKey(1000)


def feature_selection(x, y, feature_names):
    selector = SelectKBest(mutual_info_classif, k=20).fit(x, y)
    print("\n\nK-Best with Mutual Information:")
    print("\n".join(feature_names[np.where(selector.get_support() == True)]))
    string = ""
    for idx, score in enumerate(selector.scores_):
        string += "%s, score: %s\n" % (feature_names[idx], score)
    print("\nFeature rankings:")
    print(string)

    best_lasso = LassoCV().fit(x, y)
    clf = Lasso(best_lasso.alpha_)
    selector = RFECV(clf, min_features_to_select=20, step=1, cv=5, n_jobs=5)
    selector = selector.fit(x, y)
    print("Lasso with alpha {} best 20 features:".format(best_lasso.alpha_))
    print("\n".join(feature_names[np.where(selector.support_ == True)]))
    string = ""
    for idx, ranking in enumerate(selector.ranking_):
        string += "%s, ranking: %s\n" % (feature_names[idx], ranking)
    print("\nFeature rankings:")
    print(string)

    clf = LinearSVC()
    selector = RFECV(clf, min_features_to_select=20, step=1, cv=5, n_jobs=5)
    selector = selector.fit(x, y)
    print("Linear SVC with default parameters best 20 features:")
    print("\n".join(feature_names[np.where(selector.support_ == True)]))
    string = ""
    for idx, ranking in enumerate(selector.ranking_):
        string += "%s, ranking: %s\n" % (feature_names[idx], ranking)
    print("\nFeature rankings:")
    print(string)


def test_models_cross_dataset(dataset_names, dataset_paths, mat_paths, balanceds, save_imagess, scoring=None,
                              selected_features="all"):
    features_list, labels_list, feature_names_list, selected_pixels_list, paths_list = ml_utils.create_multidataset_samples(
        dataset_names, dataset_paths, mat_paths, balanceds, save_imagess)

    feature_names = feature_names_list[0]
    if selected_features == "all":
        selected_features = ";".join(feature_names)
    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features.split(";")]

    selected_features = "*%s" % selected_features.replace(";", "\n*")
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    # classifiers = [LinearSVC(class_weight='balanced'),
    classifiers = [LinearSVC(),
                   Lasso(alpha=LassoCV().fit(features_list[0][:, selected_indices], labels_list[0]).alpha_)]
    # if scoring is None:
    #     scores = [None for clf in classifiers]
    # else:
    #     scores = scoring
    scores = [matthews_corrcoef, None]

    with open("results.txt", "w") as f:
        f.write(log_string)
        print('Training on "%s", testing on "%s"' % (dataset_names[0], dataset_names[1]))
        f.write('Training on "%s", testing on "%s"\n' % (dataset_names[0], dataset_names[1]))
        for idx, clf in enumerate(classifiers):
            print("Classifier: %s" % str(clf))
            f.write("\nClassifier: %s\n" % str(clf))
            score, metric = ml_utils.cross_dataset_validation(clf, features_list[0][:, selected_indices],
                                                              labels_list[0],
                                                              features_list[1][:, selected_indices], labels_list[1],
                                                              selected_pixels_list[1], paths_list[1],
                                                              str(type(clf))[16:-2], scores[idx])
            print("{}: {:.2f}%".format(metric, 100 * score))
            f.write("{}: {:.2f}%\n".format(metric, 100 * score))


def train_models(x, y, feature_names, selected_pixels, paths, scoring=None, selected_features="all"):
    if selected_features == "all":
        selected_features = ";".join(feature_names)
    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features.split(";")]

    selected_features = "*%s" % selected_features.replace(";", "\n*")
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    # classifiers = [LinearSVC(class_weight='balanced'), Lasso(alpha=LassoCV().fit(x[:, selected_indices], y).alpha_)]
    classifiers = [LinearSVC()]
    # if scoring is None:
    #     scores = [None for clf in classifiers]
    # else:
    #     scores = scoring
    # scoring = [matthews_corrcoef, None]
    scoring = [matthews_corrcoef, None]

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf.get_n_splits(y)
    folds = []
    for train_index, test_index in kf.split(y):
        folds.append((train_index, test_index))
    del kf

    with open("results.txt", "w") as f:
        f.write(log_string)
        for idx, clf in enumerate(classifiers):
            print("Classifier: %s" % str(clf))
            f.write("\nClassifier: %s\n" % str(clf))
            scorer = make_scorer(scoring[idx]) if scoring[idx] is not None else None
            cv_results = cross_validate(clf, x[:, selected_indices], y, scoring=scorer, verbose=50, n_jobs=4,
                                        cv=folds, return_estimator=True)
            scores = cv_results["test_score"]
            try:
                scorer_name = " " + scoring[idx].__name__
            except AttributeError:
                scorer_name = ""
            print(str(scores))
            f.write("%s\n" % str(scores))
            print("Average{}: {:.2f}%, Std: {:.2f}%".format(scorer_name, 100 * np.mean(scores), 100 * np.std(scores)))
            f.write("Average{}: {:.2f}%, Std: {:.2f}%\n".format(scorer_name, 100 * np.mean(scores), 100 * np.std(scores)))
            predictions = ml_utils.cross_validate_predict(x[:, selected_indices], folds, cv_results)
            ml_utils.save_visual_results(selected_pixels, predictions, y, paths, str(type(clf))[16:-2])


dataset_name = "cfd-pruned"
dataset_folder = "/media/winbuntu/databases/CrackForestDatasetPruned"
mat_path = "cfd-pruned_1_to_10_weighted.mat"
# mat_path = None
balanced = -10
save_images = False

x, y, feature_names, selected_pixels, paths = ml_utils.create_samples(dataset_name, dataset_folder, mat_path,
                                                                      balanced, save_images)
selected_features = "Frangi's vesselness;Bottom-hat;Cross bottom-hat;Sliding mean 50x50;Sliding std 50x50;Sliding median 50x50;Sliding mad 50x50"
selected_features = "all"
train_models(x, y, feature_names, selected_pixels, paths, selected_features=selected_features)

# dataset_name_1 = "cfd-pruned"
# dataset_folder_1 = "/media/winbuntu/databases/CrackForestDatasetPruned"
# mat_path_1 = "cfd-pruned_1_to_10_weighted.mat"
# # mat_path_1 = None
# balanced_1 = -10
# save_images_1 = False
#
# dataset_name_2 = "cfd-pruned"
# dataset_folder_2 = "/media/winbuntu/databases/CrackForestDatasetPruned"
# mat_path_2 = "cfd-pruned.mat"
# # mat_path_2 = None
# balanced_2 = False
# save_images_2 = False
#
# test_models_cross_dataset([dataset_name_1, dataset_name_2], [dataset_folder_1, dataset_folder_2],
#                           [mat_path_1, mat_path_2],
#                           [balanced_1, balanced_2], [save_images_1, save_images_2], selected_features=selected_features)
