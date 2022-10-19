import numpy as np


class Node:
    def __init__(self, feature=None,
                 threshold=None, left=None,
                 right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


def entropy(y):
    proportions = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])


def get_split(X: np.ndarray, thresh: float):
    left_idx = np.argwhere(X <= thresh).flatten()
    right_idx = np.argwhere(X > thresh).flatten()
    return left_idx, right_idx


def information_gain(X: np.ndarray, y: np.ndarray, thresh: float):
    parent_loss = entropy(y)
    left_idx, right_idx = get_split(X, thresh)
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)

    if n_left == 0 or n_right == 0:
        return 0

    child_loss = (n_left / n) * entropy(y[left_idx]) + (n_right / n) * entropy(y[right_idx])
    return parent_loss - child_loss


def get_best_split(X: np.ndarray, y: np.ndarray, features):
    split = {'score': - 1, 'feature': None, 'thresh': None}
    for feature in features:
        x_features = X[:, feature]
        thresholds = np.unique(x_features)
        for thresh in thresholds:
            score = information_gain(x_features, y, thresh)
            if score > split['score']:
                split['score'] = score
                split['feature'] = feature
                split['thresh'] = thresh
    return split


class DecisionTree:
    def __init__(self, max_depth: int = 100, min_samples_split: int = 2):
        self.root = None
        self.n_samples = None
        self.n_features = None
        self.n_class_labels = None
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split

    def _finished(self, depth: int):
        return (depth >= self.max_depth
                or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split)

    def _build(self, X: np.ndarray, y: np.ndarray, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
        if self._finished(depth):
            most_common_label_index = np.argmax(np.bincount(y))
            return Node(value=most_common_label_index)

        random_features = np.random.choice(self.n_features, self.n_features, replace=False)
        best_spit = get_best_split(X, y, random_features)
        best_feature, best_thresh = best_spit['feature'], best_spit['thresh']

        left_idx, right_idx = get_split(X[:, best_feature], best_thresh)
        left_child = self._build(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feature, best_thresh, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        return self._traverse_tree(x, node.left) \
            if x[node.feature] <= node.threshold \
            else self._traverse_tree(x, node.right)

    def fit(self, x, y):
        self.root = self._build(x, y,)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
