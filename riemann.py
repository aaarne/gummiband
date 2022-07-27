import numpy as np


def Euclidean(x):
    if len(x.shape) == 1:
        return np.eye(2)
    elif len(x.shape) == 2:
        m = np.zeros((x.shape[0], 2, 2))
        m[:, 0, 0] = 1
        m[:, 1, 1] = 1
        return m
    else:
        raise ValueError


def compute_length(start, end, metric, resolution=1e-3, steps=None):
    """
    Compute length of straight (Euclidean straightness) line segments given a Riemannian metric
    """
    d = end - start

    if metric is None or metric.is_euclidean:
        return np.linalg.norm(d)

    def finite_sum(n):
        vector = d / n
        points = np.array([start + t * d for t in np.linspace(0, 1, int(n))])
        elements = np.einsum("i,nij,j->n", vector, metric(points), vector)
        elements[elements < 0] = 0.0
        return np.sum(np.sqrt(elements))

    if steps:
        return finite_sum(steps)
    elif resolution and np.isfinite(resolution):
        steps = np.ceil(np.linalg.norm(d) / resolution)
        return finite_sum(steps)
    else:
        m = metric(start + 0.5 * d)
        return np.sqrt(d.transpose() @ m @ d)
