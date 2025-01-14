import unittest

import numpy as np
from numpy.testing import assert_allclose


def ordinal_variation_l2(p):
    """
    Blair’s measure of ordinal variation
    :param p: Probability distribution
    :return: consensus [0,1] -  dispersal is defined as 1 minus consensus
    """
    if isinstance(p, list):
        p = np.array(p)
    L = len(p)
    # Value at mode
    fs = (np.cumsum(p[:-1]) - 0.5) ** 2
    norm = (L - 1) / 4
    return np.sum(fs) / norm


class OrdinalVariationTests(unittest.TestCase):
    def test_consensus_complete_disagreement_l1(self):
        cons = ordinal_variation_l2([0.5, 0, 0, 0, 0.5])
        self.assertEqual(0, cons)

    def test_consensus_full_agreement_l1(self):
        cons = ordinal_variation_l2([1, 0, 0, 0, 0])
        self.assertEqual(1, cons)

    def test_consensus_full_agreement2_l1(self):
        cons = ordinal_variation_l2([0, 0, 0, 0, 1])
        self.assertEqual(1, cons)

    def test_consensus_invariance1(self):
        o1 = ordinal_variation_l2([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = ordinal_variation_l2([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)

    def test_ordinal_consensus_invariance2(self):
        o1 = ordinal_variation_l2([0.2, 0.3, 0.1, 0.4])
        o2 = ordinal_variation_l2([0.4, 0.1, 0.3, 0.2])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)
