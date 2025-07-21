import numpy as np
from scipy.differentiate import derivative
from itertools import product


def tensor_indices(n, order):
    return product(*[range(n) for _ in range(order)])


def one_hot_vector(idx, n):
    return np.array([int(k == idx) for k in range(n)])


class Metric:
    def __call__(self, x):
        raise NotImplementedError

    def christoffel_first_kind(self, x):
        raise NotImplementedError

    def christoffel_second_kind(self, x):
        raise NotImplementedError

    @property
    def is_euclidean(self):
        return False


class _Euclidean(Metric):
    def __call__(self, x):
        n = len(x)
        return np.eye(n)

    def christoffel_first_kind(self, x):
        n = len(x)
        return np.zeros((n, n, n))

    def christoffel_second_kind(self, x):
        n = len(x)
        return np.zeros((n, n, n))

    @property
    def is_euclidean(self):
        return True


Euclidean = _Euclidean()


class NumericalMetric(Metric):
    def __init__(self, metric, spacing=1e-2):
        self._g = metric
        self._dx = spacing

    def __call__(self, x):
        return self._g(x)

    def inv(self, x):
        return np.linalg.inv(self(x))

    def _christoffel_second_kind_element(self, x, k, i, j):
        n = len(x)
        g = self._g
        ginv = np.linalg.inv(g(x))
        spacing = self._dx

        def compute(k, i, j):
            for s in range(n):
                a = derivative(lambda y: g(x + y * one_hot_vector(j, n))[s, i], 0, dx=spacing)
                b = derivative(lambda y: g(x + y * one_hot_vector(i, n))[s, j], 0, dx=spacing)
                c = derivative(lambda y: g(x + y * one_hot_vector(s, n))[i, j], 0, dx=spacing)
                yield .5 * ginv[k, s] * (a + b - c)

        return sum(compute(k, i, j))

    def christoffel_second_kind(self, x):
        n = len(x)
        gamma = np.empty((n, n, n))

        for k, i, j in tensor_indices(n, 3):
            gamma[k, i, j] = self._christoffel_second_kind_element(x, k, i, j)

        return gamma

    def christoffel_first_kind(self, x):
        return np.einsum("cd,dab->cab", self(x), self.christoffel_second_kind(x))

    def riemann_curvature_tensor(self, x):
        """returs R[l,i,j,k] = R^l_ijk"""
        n = len(x)
        R = np.empty((n, n, n, n))
        gamma_second = self.christoffel_second_kind(x)
        gamma_fun = self._christoffel_second_kind_element

        def del_gamma(n1, n2, n3, d):
            return derivative(lambda y: gamma_fun(x + y * one_hot_vector(d, n), n1, n2, n3), 0, dx=self._dx)

        for l, i, j, k in tensor_indices(n, 4):
            R[l, i, j, k] = del_gamma(l, i, k, j) - del_gamma(l, j, k, i)

        return R + np.einsum("pik,ljp->lijk", gamma_second, gamma_second) \
               - np.einsum("pjk,lip->lijk", gamma_second, gamma_second)

    def ricci_curvature_tensor(self, x):
        return np.einsum("jijk->ik", self.riemann_curvature_tensor(x))

    def scalar_curvature(self, x):
        return np.einsum("ik,ik", self.inv(x), self.ricci_curvature_tensor(x))

    def valid(self, x):
        m = self(x)
        return np.all(np.linalg.eigvals(m) > 0)

    def gaussian_curvature(self, x):
        if len(x) == 2:
            if self.valid(x):
                return self.scalar_curvature(x)/2
            else:
                return np.nan
        else:
            ValueError("Gaussian curvature is only implemented for 2-manifolds.")