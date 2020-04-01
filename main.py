import cv2
import numpy as np

import data
import utils

from math import ceil

training_image_paths, ground_truth_image_paths = data.paths_generator_cfd("/media/winbuntu/databases/CrackForest-dataset/image")
x_cfd, y_cfd = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
training_image_paths, ground_truth_image_paths = data.paths_generator_crack_dataset("/media/winbuntu/databases/CrackDataset/TITS/IMAGES/AIGLE_RN")
x_aigle, y_aigle = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
training_image_paths, ground_truth_image_paths = data.paths_generator_crack_dataset("/media/winbuntu/databases/CrackDataset/TITS/IMAGES/ESAR")
x_esar, y_esar = data.images_from_paths(training_image_paths), data.images_from_paths(ground_truth_image_paths)
x = x_cfd + x_aigle + x_esar
y = y_cfd + y_aigle + y_esar


for idx in range(len(x)):
    cracks, preprocessing = utils.find_cracks(x[idx])
    overlay = np.maximum((x[idx] / 2).astype(np.uint8), cracks)
    cv2.imshow("original / bh / filter_0 / filter_1", np.concatenate((x[idx], preprocessing[1], preprocessing[2], preprocessing[3]), axis=1))
    cv2.waitKey(1000)


