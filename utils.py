import cv2
import numpy as np
import statistics_and_thresholding as sat

from math import ceil, floor


def create_structuring_elements(line_length, dir_step):
    directions = [(dir_step * i) * (np.pi / 180.0) for i in range(int(floor(180.0 / dir_step)))]  # radians
    structuring_elements = []

    for direction in directions:
        width = int(line_length * np.cos(direction))
        height = int(line_length * np.sin(direction))
        se = np.zeros((max(height, 1), max(abs(width), 1)), np.uint8)
        if width > 0:
            cv2.line(se, (0, 0), (width - 1, max(height - 1, 0)), 1, 1)
        elif width < 0:
            cv2.line(se, (abs(width) - 1, 0), (0, max(height - 1, 0)), 1, 1)
        else:
            cv2.line(se, (0, 0), (0, height - 1), 1, 1)
        structuring_elements.append(se)
    return structuring_elements


def filter_by_length(img, length=None):
    if length is None:
        length = int((img.shape[0] + img.shape[1]) / 20)
    cracks = np.zeros(img.shape, dtype=np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        perimeter = cv2.arcLength(c, False)
        if perimeter > length:
            cv2.drawContours(cracks, [c], -1, 1, -1)
    return cracks * img


def filter_by_shape(img, circularity_threshold=0.5):
    cracks = np.zeros(img.shape, dtype=np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / perimeter ** 2
        if circularity < circularity_threshold and area > 0:
            cv2.drawContours(cracks, [c], -1, 1, -1)
    return cracks * img


def multi_dir_bottom_hat_transform(img, se_length=10, dir_step=10, open_first=0.2):
    structuring_elements = create_structuring_elements(se_length, dir_step)
    if open_first is not None:
        reduced_structuring_elements = create_structuring_elements(max(2, int(open_first*se_length)), dir_step)

    images = []
    differences = []
    for se, se2 in zip(structuring_elements, reduced_structuring_elements):
        if open_first is not None:
            resulting_image = cv2.morphologyEx(
                cv2.morphologyEx(img, cv2.MORPH_OPEN, se2), cv2.MORPH_CLOSE, se)
        else:
            resulting_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
        images.append(resulting_image)
        differences.append(resulting_image-img)
    supremum = images[0]
    for closed_image in images[1:]:
        supremum = np.maximum(supremum, closed_image)
    return min_max_contrast_enhancement(supremum.astype(np.int16) - img.astype(np.int16))


def multi_dir_linear_filtering(img, kernel_length=10, dir_step=10):
    structuring_elements = create_structuring_elements(kernel_length, dir_step)
    for idx, se in enumerate(structuring_elements):
        structuring_elements[idx] = se / se.sum()
    filtered_images = []
    for se in structuring_elements:
        blur_image = cv2.filter2D(img, -1, se, borderType=cv2.BORDER_ISOLATED)
        filtered_images.append(blur_image)
    supremum = filtered_images[0]
    for image in filtered_images[1:]:
        supremum = np.maximum(supremum, image)
    return supremum


def min_max_contrast_enhancement(img):
    contrast_enhanced = (img.astype(np.float) - img.min()) * 255 / (img.max() - img.min())
    return contrast_enhanced.astype(np.uint8)


def morph_link_c(img, se_size):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    dilated = cv2.dilate(img, se)
    return dilated


def find_cracks(img, se_size=10, ori_step=10):
    better_contrast = min_max_contrast_enhancement(img)
    bottom_hat = multi_dir_bottom_hat_transform(better_contrast, se_size, ori_step, 0.2)
    filtered = multi_dir_linear_filtering(bottom_hat, se_size, ori_step)
    # filtered = bottom_hat
    binarized = sat.multi_dir_prob_filter_threshold(filtered, se_size, 1, 1e-10, 10)
    # binarized = sat.mad_threshold(filtered, decision_level=2.5, b=1.4826)
    dilated = morph_link_c(binarized, se_size)
    cracks = filter_by_shape(dilated)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    return clean, [better_contrast, bottom_hat, filtered, binarized]
    # return clean, [better_contrast, bottom_hat, filtered, binarized, dilated, cracks, skeleton, connected_skeleton,
    #                clean]
