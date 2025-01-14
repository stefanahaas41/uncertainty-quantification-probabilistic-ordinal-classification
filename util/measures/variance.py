import unittest

import numpy as np
from numpy.testing import assert_allclose


def mean_for_probabilities(probabilities):
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    return expected_value

def variance_for_probabilities(probabilities):
    """
    Calculate Variance of Discrete Random Variable

    :param probabilities:
    :return:
    """
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    squared_difference = (expected_value - values) ** 2
    return squared_difference @ probabilities

def standard_deviation_for_probabilities(probabilities):
    return np.sqrt(variance_for_probabilities(probabilities))

class VarianceTests(unittest.TestCase):
    def test_variance_invariance(self):
        o1 = variance_for_probabilities([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = variance_for_probabilities([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)