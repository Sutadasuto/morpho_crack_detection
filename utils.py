import cv2
import numpy as np
import sympy as sym

from math import ceil, floor
from scipy.stats import binom
from scipy.special import comb


def bottom_hat(img, structuring_elements):
    operated_images = []
    for se in structuring_elements:
        bh = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_ISOLATED)
        operated_images.append(bh)
    a = operated_images[0]
    for oi in operated_images[1:]:
        a = np.minimum(a, oi)
    return a


def create_structuring_elements(line_length, directions):

    structuring_elements = []
    for direction in directions:
        width = int(line_length * np.cos(direction))
        height = int(line_length * np.sin(direction))
        se = np.zeros((max(height, 1), max(abs(width), 1)), np.uint8)
        if width > 0:
            cv2.line(se, (0, 0), (width-1, max(height-1, 0)), 1, 1)
        elif width < 0:
            cv2.line(se, (abs(width)-1, 0), (0, max(height-1, 0)), 1, 1)
        else:
            cv2.line(se, (0, 0), (0, height-1), 1, 1)
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


def sliding_mod_bottom_hat(img, line_length, step=45.0):
    window_size = (line_length*10, line_length*10)
    anchor_o = [0, 0]

    cols = ceil(img.shape[1] / window_size[1])
    rows = ceil(img.shape[0] / window_size[0])
    windows = []
    for row in range(rows):
        for col in range(cols):
            windows.append(img[row*window_size[0]:(row+1)*window_size[0], col*window_size[1]:(col+1)*window_size[1]])

    for window in windows:
        # L, filtered, binarized = mod_bottom_hat(window, line_length, step)

        ###
        bottom_hat, filtered, binarized_o = mod_bottom_hat(window, line_length, step)
        dilated = morph_link_c(binarized_o, ceil(line_length / 20))
        cracks = filter_by_shape(dilated)
        skeleton = cv2.ximgproc.thinning(cracks)
        connected_skeleton = morph_link_c(skeleton, ceil(line_length / 20))
        clean = filter_by_length(connected_skeleton)
        ###

        cv2.imshow("or / bh", np.concatenate((window, bottom_hat), axis=1))
        cv2.waitKey(2000)
    a= 0



def mod_bottom_hat(img, line_length, step=45.0):
    # step in degrees
    directions = [(step*i)*(np.pi/180.0) for i in range(int(floor(180.0/step)))]  # radians
    structuring_elements = create_structuring_elements(line_length, directions)
    norm = (img.astype(np.float) - img.min()) * 255/(img.max() - img.min())
    norm = norm.astype(np.uint8)
    a = opening_closing(norm, structuring_elements)
    # b = bottom_hat(norm, structuring_elements)
    L = a.astype(np.int16) - norm.astype(np.int16)
    L = (255 * (L.astype(np.float32) - L.min()) / (L.max() - L.min())).astype(np.uint8)
    # L = b.astype(np.int16) - norm.astype(np.int16)
    # cv2.imshow("L", np.concatenate(()))
    # cv2.waitKey(10000)
    # M = np.maximum(np.zeros(L.shape, np.int16), L - int(L.mean() + 0*L.std())).astype(np.uint8)
    # L = (M - M.min()) / (M.max() - M.min())
    # L = (255*L).astype(np.uint8)

    # hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    # ret, binarized = cv2.threshold(L, min(254, int(L.mean() + 2*L.std())), 255, cv2.THRESH_BINARY)
    filtered_0 = multi_or_conv(L, 10, 10, 1, 1, 0)
    filtered_1 = multi_or_prob_conv(filtered_0, 10, 1, 1e-10, 10)
    ret, binarized = cv2.threshold(filtered_1, min(254, int(filtered_1.mean() + 2*filtered_1.std())), 255, cv2.THRESH_BINARY)
    return L, filtered_0, filtered_1, binarized


def conv(img, kernel, iterations=5, t=10000):
    if iterations == 0:
        return img
    else:
        img = cv2.filter2D(conv(img, kernel, iterations-1, t), -1, kernel, borderType=cv2.BORDER_ISOLATED)
        if t > 0:
            cv2.imshow("filtered", img)
            cv2.waitKey(t)
        return img


def multi_or_conv(img, length=10, step=10, iterations=5, iterations_per_filter=5, t=0):
    if iterations == 0:
        return img
    else:
        img = multi_or_conv(img, length, step, iterations-1, iterations_per_filter, t)
        directions = [(step * i) * (np.pi / 180.0) for i in range(int(floor(180.0 / step)))]  # radians
        structuring_elements = create_structuring_elements(length, directions)
        for idx, se in enumerate(structuring_elements):
            structuring_elements[idx] = se/se.sum()
        images = []
        for se in structuring_elements:
            ri = conv(img, se, iterations_per_filter, t)
            ret, binarized = cv2.threshold(ri, 125, 255, cv2.THRESH_BINARY)
            binarized = ri
            images.append(binarized)
        m = images[0]
        for fi in images[1:]:
            m = np.maximum(m, fi)
        return m


def multi_or_prob_conv(img, length=10, r=0.9, epsilon=0.01, step=10):
    directions = [(step * i) * (np.pi / 180.0) for i in range(int(floor(180.0 / step)))]  # radians
    structuring_elements = create_structuring_elements(length, directions)
    probabilities = np.array([0 for i in range(256)], dtype=np.float32)
    unique, counts = np.unique(img, return_counts=True)
    for idx, intensity in enumerate(unique):
        probabilities[intensity] = counts[idx] / (img.shape[0] * img.shape[1])
    accumulated_prob = np.array([0 for i in range(256)], dtype=np.float32)
    prev_prob = 0
    for intensity in range(probabilities.shape[0]):
        accumulated_prob[intensity] = probabilities[intensity] + prev_prob
        prev_prob = accumulated_prob[intensity]

    images = []
    # threshold = int(img.mean() + 2*img.std())
    # ret, binarized = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # binarized = (binarized/255).astype(np.float32)
    # p = binarized.sum() / float(img.shape[0] * img.shape[1])
    # binary_distribution = [binom(length, p).pmf(k) for k in range(length+1)]
    # accumulated_prob = [binary_distribution[0]]
    # for idx, prob in enumerate(binary_distribution[1:], 1):
    #     accumulated_prob.append(accumulated_prob[idx-1] + binary_distribution[idx])
    # for idx, prob in enumerate(accumulated_prob):
    #     if prob >= binary_prob_thres:
    #         min_k = idx
    #         break
    # max_p = max(binary_distribution)
    # min_p = min(binary_distribution)
    for se in structuring_elements:
        # n_possible_locations = (img.shape[0] - se.shape[0]) * (img.shape[1] - se.shape[1])
        l = int(se.sum())
        k = int(r*l)
        p = sym.Symbol('p')
        probabilities = [comb(l, i)*(p**i)*(1-p)**(l-i) for i in range(k, l+1)]
        equation = -epsilon
        for prob in probabilities:
            equation += prob #* n_possible_locations
        max_p = 0
        solutions = sym.solveset(equation, p).args
        for solution in solutions:
            if type(solution) is sym.numbers.Float:
                if max_p < float(solution) < 1.0:
                    max_p = solution
        for intensity, prob in enumerate(accumulated_prob):
            if (1 - prob) <= max_p:
                threshold = intensity
                break
        ret, binarized = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        binarized = binarized.astype(np.float32)
        filtered = cv2.filter2D(binarized, -1, se, borderType=cv2.BORDER_ISOLATED)
        rescaled = np.maximum(np.zeros(filtered.shape), (filtered - (k-1)) / (l - (k-1)))
        rescaled = (255*rescaled).astype(np.uint8)
        # for i in range(rescaled.shape[0]):
        #     for j in range(rescaled.shape[1]):
        #         intensity = filtered[i,j]
        #         # pix_p = binary_distribution[int(intensity)]
        #         # rescaled[i, j] = (max_p - pix_p) / (max_p - min_p)
        #         if intensity >= min_k:
        #             rescaled[i, j] = 255
        images.append(rescaled)
    m = images[0]
    for fi in images[1:]:
        m = np.maximum(m, fi)
    return m


def morph_link_c(img, se_size):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    # close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    # open = cv2.morphologyEx(close, cv2.MORPH_OPEN, se)
    dilated = cv2.dilate(img, se)
    return dilated


def opening_closing(img, structuring_elements):
    operated_images = []
    for se in structuring_elements:
        bh = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, se, borderType=cv2.BORDER_ISOLATED), cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_ISOLATED)
        operated_images.append(bh)
    a = operated_images[0]
    for oi in operated_images[1:]:
        a = np.maximum(a, oi)
    return a


def find_cracks(img, se_size=10, ori_step=10):
    # bottom_hat, filtered, binarized_o = sliding_mod_bottom_hat(img, se_size, ori_step)
    bottom_hat, filtered_0, filtered_1, binarized_o = mod_bottom_hat(img, se_size, ori_step)
    dilated = morph_link_c(binarized_o, se_size)
    # dilated = morph_link_c(filtered, ceil(se_size / 20))
    cracks = filter_by_shape(dilated)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    # clean = connected_skeleton
    return clean, [bottom_hat, filtered_0, filtered_1, binarized_o, dilated, skeleton, connected_skeleton]