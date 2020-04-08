import cv2
import numpy as np
import sympy as sym
import utils
from scipy.special import comb
from math import floor, ceil


def calculate_median_and_mad(img, b=1.4826):
    #   Read next article for further reference:
    #   C. Leys, C. Ley, O. Klein, P. Bernard, and L. Licata, “Detecting outliers: Do not use standard deviation around
    #   the mean, use absolute deviation around the median,” Journal of Experimental Social Psychology, vol. 49, no. 4,
    #       pp. 764–766, Jul. 2013, doi: 10.1016/j.jesp.2013.03.013.
    freq_hist = get_frequency_histogram(img)
    freq_hist = np.concatenate(
        (freq_hist[1:][::-1], freq_hist))  # Mirror histogram over y axis to get a normal-like distribution
    unrolled_hist = []
    for intensity in range(len(freq_hist)):
        unrolled_hist += [intensity - 255 for i in range(freq_hist[intensity])]
    n_pixels = len(unrolled_hist)
    m_pos, r = divmod(n_pixels + 1, 2)
    m_pos -= 1  # Python counts from 0, not from 1
    if r == 0:
        m_j = unrolled_hist[m_pos]
    else:
        m_j = (unrolled_hist[m_pos] + unrolled_hist[m_pos + 1]) / 2
    abs_median_deviation = sorted(abs(np.array(unrolled_hist, dtype=np.float64) - m_j))
    if r == 0:
        m_i = abs_median_deviation[m_pos]
    else:
        m_i = (abs_median_deviation[m_pos] + abs_median_deviation[m_pos + 1]) / 2
    return m_j, b * m_i


def mad_threshold(img, decision_level=2.5, b=1.4826):
    median, mad = calculate_median_and_mad(img, b)
    threshold = median + decision_level * mad
    ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def get_accumulated_frequency(frequency_histogram):
    accumulated_frequency = np.zeros(frequency_histogram.shape, dtype=np.uint16)
    prev_freq = 0
    for intensity in range(frequency_histogram.shape[0]):
        accumulated_frequency[intensity] = prev_freq + frequency_histogram[intensity]
        prev_freq = accumulated_frequency[intensity]
    return accumulated_frequency


def get_accumulated_probability(probability_histogram):
    accumulated_probability = np.zeros(probability_histogram.shape, dtype=np.float64)
    prev_prob = 0
    for intensity in range(probability_histogram.shape[0]):
        accumulated_probability[intensity] = prev_prob + probability_histogram[intensity]
        prev_prob = accumulated_probability[intensity]
    return accumulated_probability


def get_frequency_histogram(img):
    intensity_frequencies = np.array([0 for i in range(256)], dtype=np.uint64)
    unique_values, counts = np.unique(img, return_counts=True)
    for idx, intensity in enumerate(unique_values):
        intensity_frequencies[intensity] = counts[idx]
    return intensity_frequencies


def get_probability_histogram(img):
    n_pixels = img.shape[0] * img.shape[1]
    intensity_frequencies = get_frequency_histogram(img)
    intensity_probabilities = intensity_frequencies.astype(np.float64) / n_pixels
    return intensity_probabilities


def multi_dir_prob_filter_threshold(img, kernel_length=10, r=1, epsilon=0.01, dir_step=10):
    structuring_elements = utils.create_structuring_elements(kernel_length, dir_step)
    prob_histogram = get_probability_histogram(img)
    accumulated_prob = get_accumulated_probability(prob_histogram)

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
        # The more connected pixels, the greater the crack probability. We filter for minimum k pixels
        rescaled = np.maximum(np.zeros(filtered.shape), (filtered.astype(np.float32) - (k - 1)) / (l - (k - 1)))
        rescaled = (255 * rescaled).astype(np.uint8)
        images.append(rescaled)
    supremum = images[0]
    for image in images[1:]:
        supremum = np.maximum(supremum, image)
    ret, binary = cv2.threshold(supremum, 1, 255, cv2.THRESH_BINARY)
    return binary


def multi_dir_prob_filter_threshold_2(img, kernel_length=10, r=1, epsilon=0.01, dir_step=10):
    structuring_elements = utils.create_structuring_elements(kernel_length, dir_step)
    prob_histogram = get_probability_histogram(img)
    accumulated_prob = get_accumulated_probability(prob_histogram)

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
        # The more connected pixels, the greater the crack probability. We filter for minimum k pixels
        rescaled = np.maximum(np.zeros(filtered.shape), (filtered.astype(np.float32) - (k - 1)) / (l - (k - 1)))
        rescaled = (255 * rescaled).astype(np.uint8)
        cracks = np.zeros(rescaled.shape, dtype=np.uint8)
        se_height, se_width = se.shape
        for i in range(rescaled.shape[0]):
            for j in range(rescaled.shape[1]):
                intensity = rescaled[i, j]
                if intensity > 0:
                    cracks[i - floor(se_height / 2): i + ceil(se_height / 2),
                    j - floor(se_width / 2): j + ceil(se_width / 2)] = np.maximum(
                        cracks[i - floor(se_height / 2): i + ceil(se_height / 2),
                        j - floor(se_width / 2): j + ceil(se_width / 2)], intensity * se)
        images.append(cracks)
    supremum = images[0]
    for image in images[1:]:
        supremum = np.maximum(supremum, image)
    ret, binary = cv2.threshold(supremum, 1, 255, cv2.THRESH_BINARY)
    return binary
