import unittest

import numpy as np
from numpy.testing import assert_allclose


def ordinal_consensus(p):
    """
    Leik’s measure of ordinal dispersal
    :param p: Probability distribution
    :return: disperal [0,1] - consensus is defined as 1 minus dispersal
    """
    if isinstance(p, list):
        p = np.array(p)

    fs = np.cumsum(p)

    def func(x):
        if x < .5:
            return x
        else:
            return 1 - x

    func_vec = np.vectorize(func)
    d = func_vec(fs)
    D = 2 * np.sum(d) / (p.size - 1)
    return D


class OrdinalConsensusTests(unittest.TestCase):
    def test_ordinal_consensus_invariance1(self):
        o1 = ordinal_consensus([0.1, 0.3, 0.4, 0.15, 0.05])  # 0.1,0.4,0.8,0.95 --> 0.4,0.1,0.3,0.45
        o2 = ordinal_consensus([0.05, 0.15, 0.4, 0.3, 0.1])  # 0.05,0.2,0.6,0.9 -->0.45,0.3,0.1,0.4
        assert_allclose(o1, o2, rtol=1e-7, atol=0)

    def test_ordinal_consensus_invariance2(self):
        o1 = ordinal_consensus([0.2, 0.3, 0.1, 0.4])  # 0.2,0.5,0.6  --> 0.3,0,0.1
        o2 = ordinal_consensus([0.4, 0.1, 0.3, 0.2])  # 0.4,0.5,0.8  --> 0.1,0,0.3
        assert_allclose(o1, o2, rtol=1e-7, atol=0)
