import unittest

import numpy as np
from numpy.testing import assert_allclose


def coefficient_of_agreement_std(p):
    return (1 + coefficient_of_agreement(p)) / 2


def coefficient_of_agreement(p):
    if isinstance(p, list):
        p_temp = np.array(p)
    else:
        p_temp = p.copy()

    K = len(p)
    AA = 0
    for i in range(K):
        pattern = np.where(p_temp > 0, 1, 0)
        if max(p_temp) == 0:
            break
        tu, tdu = get_triplets(pattern)
        if tu == 0 and tdu == 0:
            U_layer = 1
        else:
            U_layer = ((K - 2) * tu - (K - 1) * tdu) / \
                      ((K - 2) * (tu + tdu))
        A_layer = U_layer * (1 - ((sum(pattern) - 1) / (K - 1)))
        m = np.min(p_temp[p_temp > 0])
        L = pattern * m
        w = sum(L)
        AA += w * A_layer
        p_temp = p_temp - m
    return AA


def coefficient_of_agreement_old(p):
    """
    Van der Eijk's coefficient of agreement
    -1 :    Complete disagreement
    0:      Uniform
    1 :     Complete agreement
    :param p: probability distribution
    :return: [-1,1]
    """
    if isinstance(p, list):
        p = np.array(p)
    else:
        p = p.copy()

    num_categories = len(p)
    # Decompose distribution into layers
    i = 0
    patterns = np.empty((0, num_categories), int)
    weights = np.array([])
    while (True):
        min_prob = min(p[p > 0])
        pattern = np.where(p > 0, 1, 0)
        weights = np.append(weights, sum(pattern) * min_prob)
        patterns = np.append(patterns, [pattern], axis=0)
        p[p > 0] -= min_prob
        if len(p[p > 0]) == 0:
            break
        i += 1

    # Aggregate layers
    A = 0
    for pattern in patterns:
        # Get all triplets in layer pattern
        tu, tdu = get_triplets(pattern)
        if tu == 0 and tdu == 0:
            U_layer = 1
        else:
            U_layer = ((num_categories - 2) * tu - (num_categories - 1) * tdu) / \
                      ((num_categories - 2) * (tu + tdu))

        A_layer = U_layer * (1 - ((sum(pattern) - 1) / (num_categories - 1)))
        A += weights[i] * A_layer
    return A


def get_triplets(pattern):
    tu = 0
    tdu = 0
    for i in range(len(pattern) - 2):
        for j in range(i + 1, len(pattern) - 1):
            for k in range(j + 1, len(pattern)):
                triplet = f"{int(pattern[i])}{int(pattern[j])}{int(pattern[k])}"
                # Get unimodal triplets
                if triplet == "110" or triplet == "011":
                    tu += 1
                # Get non-unimodal triplets
                elif triplet == "101":
                    tdu += 1
    return tu, tdu


class CoefficientOfAgreementTests(unittest.TestCase):
    def test_agreement1(self):
        A = coefficient_of_agreement([0.5, 0.5, 0, 0, 0])
        self.assertEqual(0.75, A)

    def test_agreement2(self):
        A = coefficient_of_agreement([0.5, 0, 0.5, 0, 0])
        self.assertEqual(0.16666666666666666, A)

    def test_agreement3(self):
        A = coefficient_of_agreement([0.05, 0.15, 0.45, 0.2, 0.15])
        self.assertEqual(0.42500000000000004, A)

    def test_uniform(self):
        A = coefficient_of_agreement([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(0, A)

    def test_full_agreement(self):
        A = coefficient_of_agreement([1, 0, 0, 0, 0])
        self.assertEqual(1, A)

    def test_complete_disagreement(self):
        A = coefficient_of_agreement([0.5, 0, 0, 0, 0.5])
        self.assertEqual(-1, A)

    def test_agreement(self):
        A = coefficient_of_agreement_std([1, 0, 0, 0, 0])
        self.assertEqual(1, A)

    def test_disagreement(self):
        A = coefficient_of_agreement_std([0.5, 0, 0, 0, 0.5])
        self.assertEqual(0, A)

    def test_uniform(self):
        A = coefficient_of_agreement_std([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(0.5, A)

    def testB(self):
        A = coefficient_of_agreement_std([0.25, 0.25, 0, 0.25, 0.25])
        self.assertEqual(0.431, round(A, 3))

    def testC(self):
        A = coefficient_of_agreement_std([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(0.5, round(A, 3))

    def testD(self):
        A = coefficient_of_agreement_std([0.19, 0.16, 0.26, 0.29, 0.1])
        self.assertEqual(0.603, round(A, 3))

    def testE(self):
        A = coefficient_of_agreement_std([0.0, 0.5, 0.0, 0.5, 0.0])
        self.assertEqual(0.583, round(A, 3))

    def testF(self):
        A = coefficient_of_agreement_std([0.1, 0.2, 0.4, 0.2, 0.1])
        self.assertEqual(0.675, round(A, 3))

    def testG(self):
        A = coefficient_of_agreement_std([0.15, 0, 0.7, 0.0, 0.15])
        self.assertEqual(0.712, round(A, 3))

    def testH(self):
        A = coefficient_of_agreement_std([0.5, 0.5, 0, 0, 0])
        self.assertEqual(0.875, round(A, 3))

    def testI(self):
        A = coefficient_of_agreement_std([0, 0.5, 0.5, 0, 0])
        self.assertEqual(0.875, round(A, 3))

    def testQ(self):
        A = coefficient_of_agreement_std([0.5, 0, 0, 0.5, 0])
        self.assertEqual(0.292, round(A, 3))

    def test_invariance(self):
        o1 = coefficient_of_agreement_std([0.1, 0.3, 0.4, 0.15, 0.05])
        o2 = coefficient_of_agreement_std([0.05, 0.15, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)

    def test_invariance_even(self):
        o1 = coefficient_of_agreement_std([0.1, 0.3, 0.4, 0.2])
        o2 = coefficient_of_agreement_std([0.2, 0.4, 0.3, 0.1])
        assert_allclose(o1, o2, rtol=1e-7, atol=0)
