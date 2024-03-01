import unittest

import numpy as np


def consensus(p):
    """
    A complete lack of consensus generates a value of 0, and a complete consensus of
    opinion yields a value of 1.
    :param p:
    :return:
    """
    if isinstance(p, list):
        p = np.array(p)

    i = np.arange(len(p)) + 1
    mu = np.sum(p * i)
    eps = 1e-25
    log_exp = p * np.log2(eps + (1 - (abs(i - mu) / (max(i) - min(i)))))
    return 1 + np.sum(log_exp)


class ConsensusTests(unittest.TestCase):
    def test_consensus_complete_disagreement(self):
        cons = consensus([0.5, 0, 0, 0, 0.5])
        self.assertEqual(0, cons)

    def test_unimodal(self):
        cons = consensus([0.1, 0.2, 0.4, 0.2, 0.1])
        self.assertEqual(0.63, round(cons, 2))

    def test_uniform(self):
        cons = consensus([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(0.43, round(cons, 2))

    def test_consensus_full_agreement1(self):
        cons = consensus([0, 0, 1, 0, 0])
        self.assertEqual(1, cons)

    def test_consensus_full_agreement2(self):
        cons = consensus([1, 0, 0, 0, 0])
        self.assertEqual(1, cons)
