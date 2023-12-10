import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from alapana_nn.utils import Util


class BaseDistanceMetric:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SqL2(BaseDistanceMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        x = np.broadcast_to(x, y.shape)
        return np.sum((x - y) ** 2, axis=-1)


class JSDivergence(BaseDistanceMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        x = np.broadcast_to(x, y.shape)

        out = np.zeros((len(y),))
        for i in range(len(y)):
            out[i] = Util.js_div(x[i], y[i])
        return out


class CompoundDistance(BaseDistanceMetric):
    def __init__(self, l2_strength=1.0):
        super().__init__()
        self.js = JSDivergence()
        self.l2 = SqL2()
        self.a = l2_strength

    def __call__(self, h_x, h_y, x0, y0):
        js_loss = self.js(h_x, h_y)
        l2_loss = self.a * self.l2(x0, y0)
        return js_loss + l2_loss


class KMeans:
    def __init__(self, k: int, distance_fn: BaseDistanceMetric or None, max_iter: int = 100, ckpt_path=None):
        self.k = k
        self.dist_fn = distance_fn
        self.max_iter = max_iter

        self.labels = None
        self.centroids = None
        self.clusters = []

        if ckpt_path is not None:
            self.load(ckpt_path)

    def fit(self, x: np.ndarray, checkpoint_path: str or None = None):
        assert self.dist_fn is not None, "Please provide a distance metric"

        choices = np.random.choice(len(x), self.k, replace=False)
        self.centroids = x[choices]

        clusters = []
        for j in range(self.max_iter):
            clusters = [[] for _ in range(self.k)]
            for i in range(len(x)):
                d = self.dist_fn(x[i], self.centroids)
                c_i = np.argmin(d)
                clusters[c_i].append(i)

            temp = np.zeros((self.k, *x.shape[1:]))
            for i in range(self.k):
                temp[i] = np.mean(x[clusters[i]], axis=0)

            d = self.dist_fn(temp, self.centroids)

            print(f"Epoch {j + 1}: {d.sum(): .8f}")
            if d.sum() < 1e-8:  # converged
                break

            self.centroids = temp[:]

        self.clusters = clusters
        self.__generate_labels()

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            centroid_path = os.path.join(checkpoint_path, "centroids.npy")
            clusters_path = os.path.join(checkpoint_path, "clusters.npy")
            np.save(centroid_path, self.centroids)
            np.save(clusters_path, np.array(self.clusters, dtype=object), allow_pickle=True)

    def __generate_labels(self):
        N = 0
        for c in self.clusters:
            N += len(c)
        self.labels = np.zeros(N, dtype=int)
        for i in range(len(self.clusters)):
            for j in self.clusters[i]:
                self.labels[j] = i

    def get_cluster_for(self, x: np.ndarray):
        d = self.dist_fn(x, self.centroids)
        min_d = np.argmin(d)
        return self.clusters[min_d]

    def load(self, checkpoint_path: str):
        centroid_path = os.path.join(checkpoint_path, "centroids.npy")
        clusters_path = os.path.join(checkpoint_path, "clusters.npy")
        self.centroids = np.load(centroid_path)
        self.clusters = list(np.load(clusters_path, allow_pickle=True))
        self.__generate_labels()
