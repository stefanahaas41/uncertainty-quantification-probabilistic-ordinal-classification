import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import entropy


def binary_ordinal_entropy(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        bin_prob = np.array([np.sum(p[:i + 1]), np.sum(p[i + 1:])])
        sum += entropy(bin_prob)
    return sum


def binary_ordinal_variance(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        sum += np.sum(p[:i + 1]) * np.sum(p[i + 1:])
    return sum


def binary_ordinal_margin(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        bin_prob = np.array([np.sum(p[:i + 1]), np.sum(p[i + 1:])])
        sum += 1 - abs(bin_prob[0] - bin_prob[1])
    return sum


class BinMarginTests(unittest.TestCase):
    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(2, binary_ord_conf)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(1.3333333333333335, binary_ord_conf)

    def test_dirac(self):
        p = [0, 1, 0]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(0, binary_ord_conf)

    def test_invariance(self):
        o1 = binary_ordinal_margin([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = binary_ordinal_margin([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)


class BinVarianceTests(unittest.TestCase):
    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0.5, binary_ord_variance)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0.4444444444444444, binary_ord_variance)

    def test_dirac(self):
        p = [0, 1, 0]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0, binary_ord_variance)

    def test_invariance(self):
        o1 = binary_ordinal_variance([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = binary_ordinal_variance([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)


class BinEntropyTests(unittest.TestCase):
    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(1.3862943611198906, bin_orinal_entropy)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(1.2730283365896256, bin_orinal_entropy)

    def test_dirac(self):
        p = [0, 1, 0]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(0, bin_orinal_entropy)

    def test_invariance(self):
        o1 = binary_ordinal_entropy([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = binary_ordinal_entropy([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)
