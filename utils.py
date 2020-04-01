import cv2
import numpy as np
import sympy as sym

from math import ceil, floor
from scipy.stats import binom
from scipy.special import comb


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


def multi_dir_bottom_hat_transform(img, se_length=10, dir_step=10, open_first=True):
    structuring_elements = create_structuring_elements(se_length, dir_step)

    images = []
    for se in structuring_elements:
        if open_first:
            resulting_image = cv2.morphologyEx(
                cv2.morphologyEx(img, cv2.MORPH_OPEN, se, borderType=cv2.BORDER_ISOLATED),
                cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_ISOLATED)
        else:
            resulting_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_ISOLATED)
        images.append(resulting_image)
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


def multi_dir_probabilistic_filtering(img, kernel_length=10, r=1, epsilon=0.01, dir_step=10):
    structuring_elements = create_structuring_elements(kernel_length, dir_step)
    intensity_probabilities = np.array([0 for i in range(256)], dtype=np.float32)
    unique_values, counts = np.unique(img, return_counts=True)
    n_pixels = img.shape[0] * img.shape[1]
    for idx, intensity in enumerate(unique_values):
        intensity_probabilities[intensity] = counts[idx] / n_pixels
    accumulated_prob = np.array([0 for i in range(256)], dtype=np.float32)
    prev_prob = 0
    for intensity in range(intensity_probabilities.shape[0]):
        accumulated_prob[intensity] = intensity_probabilities[intensity] + prev_prob
        prev_prob = accumulated_prob[intensity]

    images = []
    for se in structuring_elements:
        #  Expected Number of False Alarms choosing all possible paths of length l and orientation
        #       d in the current image if it doesn't have cracks: NFA
        #  Number of Possible Paths of length l and orientation d in the current image: NPP
        #  Minimum number of desired bright pixels in a path to be considered a crack: k
        #  Probability of randomly finding a bright pixel: p
        #  Probability of finding a path with k <= no. of bright pixels <= l, under a binomial distribution with l
        #       number of trials and p as success probability: P[crack|k,l,p]
        #  NFA = NPP * P[crack|k,l,p]
        l = int(se.sum())
        k = int(r * l)
        #  Probability of finding exactly k bright pixels in a path given a binomial distribution with probability
        #       p and l trials: P[k|l,p] = (lCk) * (p**k) * (1-p)**(l-k)
        p = sym.Symbol('p')
        #  P[crack|k,l,p] = ΣP[i|l,p] from i=k to k=l
        probabilities = [comb(l, i) * (p ** i) * (1 - p) ** (l - i) for i in range(k, l + 1)]
        #  Maximum percentage of admissible false alarms: epsilon
        #  epsilon = (false alarms) / NPP
        #  NFA = NPP * P[crack|k,l,p] <= epsilon * NPP
        #  P[crack|k,l,p] - epsilon <= 0
        equation = -epsilon
        for prob in probabilities:
            equation += prob
        #  We know k, l and epsilon, so we solve for p
        max_p = 0
        solutions = sym.solveset(equation, p).args
        for solution in solutions:
            if type(solution) is sym.numbers.Float:
                if max_p < float(solution) < 1.0:
                    max_p = solution
        #  We threshold the image for the intensity where the right side of the histogram has a probability <= p
        for intensity, prob in enumerate(accumulated_prob):
            if (1 - prob) <= max_p:
                threshold = intensity
                break
        ret, binarized = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        #  We convolve the SE over the image looking for all the possible crack segments with its same length and
        #       orientation
        filtered = cv2.filter2D(binarized, -1, se, borderType=cv2.BORDER_ISOLATED)
        # The more connected pixels, the greater the crack probability
        rescaled = np.maximum(np.zeros(filtered.shape), (filtered.astype(np.float32) - (k - 1)) / (l - (k - 1)))
        rescaled = (255 * rescaled).astype(np.uint8)
        images.append(rescaled)
    supremum = images[0]
    for image in images[1:]:
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
    bottom_hat = multi_dir_bottom_hat_transform(better_contrast, se_size, ori_step)
    filtered_0 = multi_dir_linear_filtering(bottom_hat, se_size, ori_step)
    filtered_1 = multi_dir_probabilistic_filtering(filtered_0, se_size, 1, 1e-10, 10)
    dilated = morph_link_c(filtered_1, se_size)
    cracks = filter_by_shape(dilated)
    skeleton = cv2.ximgproc.thinning(cracks)
    connected_skeleton = cv2.ximgproc.thinning(morph_link_c(skeleton, se_size))
    clean = filter_by_length(connected_skeleton)
    return clean, [better_contrast, bottom_hat, filtered_0, filtered_1, dilated, cracks, skeleton, connected_skeleton,
                   clean]
