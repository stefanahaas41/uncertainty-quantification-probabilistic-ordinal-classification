import unittest
from collections import Counter

import numpy as np


def dfu(input_data, histogram_input=True):
    """ The Distance From Unimodality measure
    :param: input_data: the data, by default the relative frequencies of ratings
    :param: histogram_input: False to compute rel. frequencies (ratings as input)
    :return: the DFU score
    """
    hist = input_data if histogram_input else to_hist(input_data)
    max_value = max(hist)
    pos_max = np.where(hist == max_value)[0][0]
    # right search
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):
        diff = hist[i + 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    for i in range(pos_max, 0, -1):
        diff = hist[i - 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    return max_diff


def to_hist(scores, bins_num=3, normed=True):
    """ Creating a normalised histogram
    :param: scores: the ratings (not necessarily discrete)
    :param: bins_num: the number of bins to create
    :param: normed: whether to normalise or not, by default true
    :return: the histogram
    """
    # not keeping the values order when bins are not created
    counts, bins = np.histogram(a=scores, bins=bins_num)
    counts_normed = counts / counts.sum()
    return counts_normed if normed else counts


SCALE10 = list(range(1, 11))


def pdf(scores, scale=SCALE10):
    """ The relative frequencies of ordinal ratings.
    :param: scores: the ratings
    :param: scale: the rating categories, by default 10-point scale
    :return: the relative frequencies
    """
    # to be used when no bins are created
    freqs = Counter(scores)
    return np.array([freqs[s] / len(scores) for s in scale])


def cpdf(scores, scale=SCALE10):
    """ Cumulative relative frequencies
    :param: scores: the ratings
    :param: scale: the rating categories, by default 10-point scale
    :return: the cumultive relative frequencies
    """
    return np.cumsum(pdf(scores, scale))


class DfuTests(unittest.TestCase):
    def test_dfu1(self):
        p = dfu(np.array([0.5, 0, 0, 0, 0.5]))
        self.assertEqual(0.5, p)

    def test_dfu2(self):
        p = dfu(np.array([0.5, 0, 0.5, 0, 0]))
        self.assertEqual(0.5, p)

    def test_dfu3(self):
        p = dfu(np.array([0.5, 0.5, 0, 0, 0]))
        self.assertEqual(0, p)

    def test_dfu4(self):
        p = dfu(np.array([0.8, 0, 0, 0, 0.2]))
        self.assertEqual(0.2, p)

    def test_dfu5(self):
        p = dfu(np.array([0.2, 0, 0, 0, 0.8]))
        self.assertEqual(0.2, p)

    def test_dfuinv1(self):
        p = dfu(np.array([0.1, 0, 0.6, 0.3, 0]))
        self.assertEqual(0.1, p)

    def test_dfuinv2(self):
        p = dfu(np.array([0, 0.3, 0.6, 0, 0.1]))
        self.assertEqual(0.1, p)

    def test_dfu_normal(self):
        p = dfu(np.array([0.2, 0.1, 0.4, 0.3]))
        self.assertEqual(0.1, p)

    def test_dfu_reversed(self):
        p = dfu(np.array([0.3, 0.4, 0.1, 0.2]))
        self.assertEqual(0.1, p)
