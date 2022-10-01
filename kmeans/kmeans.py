import random
import numpy as np
from numpy.random import uniform
from utils.math import euclidian_distance


class KMeans:
    """
    Simple KMeans implementation with numpy.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, dataset: np.ndarray):
        """
        Fit the dataset to the model.
        :param dataset: (n, m) numpy array
        :return: None
        """
        self.centroids = self.init_centroids(dataset)
        prev_centroids = None
        for _ in range(self.max_iter):
            sorted_points = [[] for _ in range(self.n_clusters)]
            for point in dataset:
                distances = euclidian_distance(point, self.centroids)
                sorted_points[np.argmin(distances)].append(point)

                prev_centroids = self.centroids
                self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
                for i, centroid in enumerate(self.centroids):
                    if np.isnan(centroid).any():
                        self.centroids[i] = prev_centroids[i]

    def predict(self, dataset: np.ndarray):
        """
        Predict the cluster of each point in the dataset.
        :param dataset: (n, m) numpy array
        :return: tuple of (n, ) array and centroid index
        """
        centroids = []
        centroid_idxs = []
        for point in dataset:
            distances = euclidian_distance(point, self.centroids)
            centroid_idx = np.argmin(distances)
            centroid_idxs.append(centroid_idx)
            centroid = self.centroids[centroid_idx]
            centroids.append(centroid)
        return centroids, centroid_idx

    def init_centroids(self, dataset):
        """
        Initialize the centroids.
        :param dataset: (n, m) numpy array
        :return: (n_clusters, m) numpy array
        """
        min_, max_ = np.min(dataset, axis=0), np.max(dataset, axis=0)
        return [uniform(min_, max_) for _ in range(self.n_clusters)]


class KMeansPlusPlus(KMeans):
    """
    KMeans++ implementation.
    """

    def init_centroids(self, dataset):
        """
        Initialize the centroids.
        :param dataset: (n, m) numpy array
        :return: (n_clusters, m) array
        """
        centroids = [random.choice(dataset)]
        for _ in range(self.n_clusters - 1):
            distances = np.sum([euclidian_distance(centroid, dataset) for centroid in centroids], axis=0)
            probabilities = distances / np.sum(distances)
            index = np.random.choice(range(len(dataset)), size=1, p=probabilities)[0]
            centroids += [dataset[index]]
        return centroids
