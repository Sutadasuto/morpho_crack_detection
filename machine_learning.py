import cv2
import ml_utils
import numpy as np
import os

from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, KFold
from sklearn.svm import LinearSVC, SVC


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


def train_models(x, y, feature_names, selected_pixels, paths):
    selected_features = "Frangi's vesselness;Bottom-hat;Cross bottom-hat;Dilation Disk diameter 3;Dilation Disk diameter 5;Dilation Disk diameter 10;Erosion Disk diameter 3;Erosion Disk diameter 5;Erosion Disk diameter 10;Closing Disk diameter 3;Closing Disk diameter 5;Closing Disk diameter 10;Opening Disk diameter 3;Opening Disk diameter 5;Opening Disk diameter 10;Closing Multi-dir Line length 3;Closing Multi-dir Line length 5;Closing Multi-dir Line length 10;Sliding mean 50x50;Sliding median 50x50;Sliding std 50x50;Sliding mad 50x50"
    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features.split(";")]

    selected_features = "*%s" % selected_features.replace(";", "\n*")
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    classifiers = [LinearSVC(), Lasso(alpha=LassoCV().fit(x, y).alpha_)]
    # classifiers = [LinearSVC()]
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf.get_n_splits(y)
    folds = []
    for train_index, test_index in kf.split(y):
        folds.append((train_index, test_index))
    del kf

    with open("results_recovery.txt", "w") as f:
        f.write(log_string)
        for clf in classifiers:
            print("Classifier: %s" % str(clf))
            f.write("\nClassifier: %s\n" % str(clf))
            cv_results = cross_validate(clf, x[:, selected_indices], y, verbose=50, n_jobs=1, cv=folds, return_estimator=True)
            scores = cv_results["test_score"]
            print(str(scores))
            f.write("%s\n" % str(scores))
            print("Average: {:.2f}%, Std: {:.2f}%".format(100*np.mean(scores), 100*np.std(scores)))
            f.write("Average: {:.2f}%, Std: {:.2f}%\n".format(100*np.mean(scores), 100*np.std(scores)))
            predictions = ml_utils.cross_validate_predict(x[:, selected_indices], folds, cv_results)
            ml_utils.save_visual_results(selected_pixels, predictions, y, paths, str(type(clf))[16:-2])


# x, y, feature_names, selected_pixels, paths = ml_utils.create_samples(dataset_name_2, dataset_folder_2, mat_path_2,
#                                                                       balanced_2, save_images_2)
# feature_selection(x, y, feature_names)
# ml_utils.reconstruct_from_selected_pixels(selected_pixels, y, paths[1])

dataset_name_1 = "esar"
dataset_folder_1 = "/media/winbuntu/databases/CrackDataset"
mat_path_1 = "esar_balanced.mat"
# mat_path_1 = None
balanced_1 = True
save_images_1 = True

x, y, feature_names, selected_pixels, paths = ml_utils.create_samples(dataset_name_1, dataset_folder_1, mat_path_1, balanced_1, save_images_1)
train_models(x, y, feature_names, selected_pixels, paths)

# train_cross_dataset([dataset_name_1, dataset_folder_2], [dataset_folder_1, dataset_folder_2], [mat_path_1, mat_path_2],
#                     [balanced_1, balanced_2], [save_images_1, save_images_2])
