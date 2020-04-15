import cv2
import ml_utils
import numpy as np
import os

from sklearn import linear_model
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_predict, cross_val_score
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

x, y, feature_names = ml_utils.create_balanced_samples("cfd", "/media/winbuntu/databases/CrackForestDataset", "cfd_balanced.mat")

selector = SelectKBest(k=20).fit(x, y)
print("\n\nK-Best with ANOVA F-value:")
print("\n".join(feature_names[np.where(selector.get_support()==True)]))
string = ""
for idx, score in enumerate(selector.scores_):
    string += "%s, score: %s\n" % (feature_names[idx], score)
print("\nFeature rankings:")
print(string)

# best_lasso = linear_model.LassoCV().fit(x, y)
# print("\n\nAlpha: %s" % best_lasso.alpha_)
# print("Coefficients:")
# print(best_lasso.coef_)
#
# clf = linear_model.Lasso(best_lasso.alpha_)
# selector = RFECV(clf, min_features_to_select=20, step=1, cv=5, n_jobs=5)
# selector = selector.fit(x, y)
# print("Lasso with alpha {} best 20 features:".format(best_lasso.alpha_))
# print("\n".join(feature_names[np.where(selector.support_==True)]))
# string = ""
# for idx, ranking in enumerate(selector.ranking_):
#     string += "%s, ranking: %s\n" % (feature_names[idx], ranking)
# print("\nFeature rankings:")
# print(string)
#
# clf = LinearSVC()
# selector = RFECV(clf, min_features_to_select=20, step=1, cv=5, n_jobs=5)
# selector = selector.fit(x, y)
# print("Linear SVC with default parameters best 20 features:")
# print("\n".join(feature_names[np.where(selector.support_==True)]))
# string = ""
# for idx, ranking in enumerate(selector.ranking_):
#     string += "%s, ranking: %s\n" % (feature_names[idx], ranking)
# print("\nFeature rankings:")
# print(string)
#
# clf = SVC()
# selector = RFECV(clf, min_features_to_select=20, step=1, cv=5, n_jobs=5)
# selector = selector.fit(x, y)
# print("RBF kernel SVC with default parameters best 20 features:")
# print("\n".join(feature_names[np.where(selector.support_==True)]))
# string = ""
# for idx, ranking in enumerate(selector.ranking_):
#     string += "%s, ranking: %s\n" % (feature_names[idx], ranking)
# print("\nFeature rankings:")
# print(string)