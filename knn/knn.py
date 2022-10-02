import numpy as np
from utils import math


def most_common(lst):
    return max(set(lst), key=lst.count)


class KNN:
    """
    Simple KNN implementation with numpy.
    """

    def __init__(self, k: int = 5, metric: str = 'euclidian'):
        self.labels = None
        self.dataset = None
        self.k = k
        if metric == 'euclidian':
            self.metric = math.euclidian_distance
        else:
            raise ValueError('Invalid metric.')

    def fit(self, dataset: np.ndarray, labels: np.ndarray):
        """
        Fit the dataset to the model.
        :param dataset: (n, m) numpy array
        :param labels: (n, ) numpy array
        :return: None
        """
        self.dataset = dataset
        self.labels = labels

    def predict(self, dataset: np.ndarray):
        """
        Predict the cluster of each point in the dataset.
        :param dataset: (n, m) numpy ndarray
        :return (n, ) array
        """
        neighbors = []
        for point in dataset:
            distances = self.metric(point, self.dataset)
            y_sorted = [y for _, y in sorted(zip(distances, self.labels))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))

    def get_accuracy(self, dataset: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the model accuracy.
        :param dataset: (n, m) numpy ndarray
        :param labels: (n, ) numpy ndarray
        :return: accuracy of the model
        """
        y_pred = self.predict(dataset)
        return np.sum(y_pred == labels) / len(labels)