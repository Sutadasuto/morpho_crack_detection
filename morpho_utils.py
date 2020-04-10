import cv2
import numpy as np
import statistics_and_thresholding as sat

from math import ceil, floor


def create_structuring_elements(line_length, dir_step):
    directions = [(dir_step * i) * (np.pi / 180.0) for i in range(int(floor(180.0 / dir_step)))]  # radians
    structuring_elements = []

    for direction in directions:
        width = int(line_length * np.cos(direction))
        if abs(width) < 1 or width % 2 == 0:  # SE must be strictly odd and symmetrical
            if width == 0:
                width = 1
            else:
                width += int(width / abs(width))
        height = int(line_length * np.sin(direction))
        if height < 1 or height % 2 == 0:  # SE must be strictly odd and symmetrical
            height += 1
        se = draw_symmetrical_line(height, width)
        structuring_elements.append(se)
    return structuring_elements


def reduce_structuring_elements(structuring_elements, reduction_factor):
    reduced_structuring_elements = []
    for se in structuring_elements:
        new_height = ceil(reduction_factor * se.shape[0])
        if new_height % 2 == 0:
            new_height += 1
        new_width = ceil(reduction_factor * se.shape[1])
        if new_width % 2 == 0:
            new_width += 1
        central_row = int((se.shape[0] - 1) / 2)
        central_col = int((se.shape[1] - 1) / 2)
        reduced_structuring_elements.append(
            se[
            central_row - floor(new_height / 2):central_row + ceil(new_height / 2),
            central_col - floor(new_width / 2):central_col + ceil(new_width / 2)
            ]
        )
    return reduced_structuring_elements


def draw_symmetrical_line(height, width):
    image = np.zeros((height, abs(width)), np.uint8)
    m = height / width
    x_s = [i for i in range(abs(width) + 1)]
    for i in range(abs(width)):
        x_lower_bound = x_s[i]
        x_greater_bound = x_s[i + 1]
        min_y = floor(abs(m) * x_lower_bound)
        max_y = ceil(abs(m) * x_greater_bound)
        image[min_y:max_y, x_s[i]] = 1
    if m < 0:
        image = np.flip(image, axis=1)
    return image


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
        reduced_structuring_elements = reduce_structuring_elements(structuring_elements, open_first)
    images = []
    differences = []
    for se, se2 in zip(structuring_elements, reduced_structuring_elements):
        if open_first is not None:
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)
            resulting_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, se)
            differences.append(resulting_image - opening)
        else:
            resulting_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
            differences.append(resulting_image - img)
        images.append(resulting_image)
    # supremum = images[0]
    # for closed_image in images[1:]:
    #     supremum = np.maximum(supremum, closed_image)
    # return min_max_contrast_enhancement(supremum.astype(np.int16) - img.astype(np.int16))
    supremum = differences[0]
    for transform in differences[1:]:
        supremum = np.maximum(supremum, transform)
    return min_max_contrast_enhancement(supremum)


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
    binarized = sat.multi_dir_prob_filter_threshold_2(filtered, se_size, 1, 1e-10, 10)
    # binarized = sat.mad_threshold(filtered, decision_level=2, b=1.4826)
    dilated = morph_link_c(binarized, se_size)
    cracks = filter_by_shape(dilated)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    return clean, [better_contrast, bottom_hat, filtered, binarized]
    # return clean, [better_contrast, bottom_hat, filtered, binarized, dilated, cracks, skeleton, connected_skeleton,
    #                clean]


def get_windows(img, window_size, sliding_steps):
    height, width = img.shape
    y_step = window_size[0] if sliding_steps[0] == -1 else sliding_steps[0]
    x_step = window_size[1] if sliding_steps[1] == -1 else sliding_steps[1]
    anchors = []
    x = y = 0
    while y < height:
        while x < width:
            anchors.append([y, x])
            x += x_step
        x = 0
        y += y_step

    windows = []
    for anchor in anchors:
        windows.append(img[anchor[0]: anchor[0] + window_size[0], anchor[1]: anchor[1] + window_size[1]])
    return windows, anchors


def join_windows(windows, anchors):
    window_height, window_width = windows[0].shape
    i = y = 0
    while True:
        last_width = windows[i].shape[1]
        i += 1
        y, x = anchors[i]
        if y > 0:
            width = anchors[i - 1][1] + last_width
            break
    height = anchors[-1][0] + windows[-1].shape[0]
    reconstructed = np.zeros((height, width), dtype=np.uint8)

    for window, anchor in zip(windows, anchors):
        reconstructed[anchor[0]: anchor[0] + window_height, anchor[1]: anchor[1] + window_width] = window
    return reconstructed


def find_cracks_sliding(img, se_size=10, ori_step=10, window_size=(100, 100), sliding_steps=(-1, -1)):
    windows, anchors = get_windows(img, window_size, sliding_steps)

    resulting_windows = []
    for window in windows:
        clean, [better_contrast, bottom_hat, filtered, binarized] = find_cracks(window, se_size, ori_step)
        resulting_windows.append([clean, better_contrast, bottom_hat, filtered, binarized])
    resulting_windows = np.array(resulting_windows)
    clean = join_windows(resulting_windows[:, 0], anchors)
    better_contrast, bottom_hat, filtered, binarized = [join_windows(resulting_windows[:, i], anchors) for i in
                                                          range(1, resulting_windows.shape[1])]
    cracks = filter_by_shape(binarized)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    return clean, [better_contrast, bottom_hat, filtered, binarized]


def find_cracks_adaptive_threshold(img, se_size=10, ori_step=10, window_size=(100, 100), sliding_steps=(-1, -1)):
    better_contrast = min_max_contrast_enhancement(img)
    bottom_hat = multi_dir_bottom_hat_transform(better_contrast, se_size, ori_step, 0.2)
    # filtered = multi_dir_linear_filtering(bottom_hat, se_size, ori_step)
    filtered = bottom_hat

    windows, anchors = get_windows(filtered, window_size, sliding_steps)
    resulting_windows = []
    for window in windows:
        # resulting_windows.append(sat.multi_dir_prob_filter_threshold_2(window, se_size, 1, 1e-10, 10))
        resulting_windows.append(sat.mad_threshold(window, decision_level=2, b=1.4826))
    binarized = join_windows(resulting_windows, anchors)
    dilated = morph_link_c(binarized, se_size)
    cracks = filter_by_shape(dilated)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    return clean, [better_contrast, bottom_hat, binarized]
