import numpy as np


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