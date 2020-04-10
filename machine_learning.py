import cv2
import ml_utils
import numpy as np
import os

from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

x, y, feature_names, file_names = ml_utils.create_images("cfd",
                                                         "/media/winbuntu/databases/CrackForestDataset",
                                                         "cfd.mat")
x, x_or_shape = ml_utils.flatten_pixels(x)
y, y_or_shape = ml_utils.flatten_pixels(y)

lasso = linear_model.Lasso()
print("Cross-validating with " + str(type(lasso)))
y_pred = cross_val_predict(lasso, x, y, cv=2, verbose=50)
del x

print("Preparing to show results.")
y, _ = ml_utils.reconstruct_from_flat_pixels(y, y_or_shape)
y_pred, _ = ml_utils.reconstruct_from_flat_pixels(y_pred, y_or_shape)

for i in range(y_or_shape[0]):
    current_result = np.concatenate((y[i, :, :], y_pred[i, :, :]), axis=1) * 255
    current_result = np.maximum(current_result, np.zeros(current_result.shape))
    current_result = np.minimum(current_result, 255*np.ones(current_result.shape))
    current_result = current_result.astype(np.uint8)
    if not os.path.exists("results ml"):
        os.makedirs("results ml")
    cv2.imwrite(os.path.join("results ml", file_names[i].replace(".mat", ".png")), current_result)
    # cv2.waitKey(1000)
