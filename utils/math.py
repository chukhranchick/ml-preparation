import numpy as np


def euclidian_distance(point, data):
    """
    Calculate the euclidian distance between a point and a dataset.
    :param point: (m, ) numpy array
    :param data: (n, m) numpy array
    :return: (n, ) numpy array of distances
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))
