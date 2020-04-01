import cv2
import numpy as np

import data
import utils

from math import ceil

# training_image_paths, ground_truth_image_paths = data.paths_generator_cfd("/media/winbuntu/databases/CrackForest-dataset/image")
# x_cfd, y_cfd = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
training_image_paths, ground_truth_image_paths = data.paths_generator_crack_dataset("/media/winbuntu/databases/CrackDataset/TITS/IMAGES/AIGLE_RN")
x_aigle, y_aigle = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
training_image_paths, ground_truth_image_paths = data.paths_generator_crack_dataset("/media/winbuntu/databases/CrackDataset/TITS/IMAGES/ESAR")
x_esar, y_esar = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
# x = x_cfd + x_aigle + x_esar
# y = y_cfd + y_aigle + y_esar
x = x_esar


for idx in range(len(x)):
    cracks, preprocessing = utils.find_cracks(x[idx])
    overlay = np.maximum((x[idx] / 2).astype(np.uint8), cracks)
    cv2.imshow("original / bh / filter_0 / filter_1", np.concatenate((x[idx], preprocessing[0], preprocessing[1], preprocessing[2]), axis=1))
    cv2.waitKey(1000)

# img_path = "/media/winbuntu/databases/CrackDataset/TITS/IMAGES/ESAR/Im_noGT_ESAR_30a.jpg"
# se_size = 100
#
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# bottom_hat, filtered, binarized_o = utils.mod_bottom_hat(img, se_size)
# dilated = utils.morph_link_c(binarized_o, ceil(se_size/20))
# cracks = utils.filter_by_shape(dilated)
# skeleton = cv2.ximgproc.thinning(cracks)
# connected_skeleton = utils.morph_link_c(skeleton, ceil(se_size/20))
# clean = utils.filter_by_length(connected_skeleton)
# cv2.imshow("bh/mlc", np.concatenate([img, bottom_hat, filtered, binarized_o, dilated, connected_skeleton, clean], axis = 1))
#
# bgr = cv2.imread(img_path)
# bgr[:, :, 2] = np.maximum(bgr[:,:, 2], clean)
# cv2.imshow("comparison", bgr)
# cv2.waitKey(60000)
