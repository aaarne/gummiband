import numpy as np
from warnings import warn
from .riemann import compute_length

import scipy.misc
from scipy.interpolate import interp1d
from scipy.spatial.kdtree import KDTree


def arclength_parametrization(line):
    diffs = np.diff(line, axis=0, prepend=line[0:1, :])
    length_elements = np.linalg.norm(diffs, axis=1)
    return np.cumsum(length_elements)


class _Curve:
    def __init__(self, line, parameters, lazy=True):
        self._points = line
        n = line.shape[1]
        self._dims = n
        self._parameters = parameters
        self._arclength = arclength_parametrization(line)
        if lazy:
            self._ints = None
            self._kdtt = None
        else:
            self._ints = self._create_interpolators()
            self._kdtt = self._create_kdtree()

    @property
    def _interpolators(self):
        if self._ints is None:
            self._ints = self._create_interpolators()
        return self._ints

    def override_points(self, new_points):
        assert new_points.shape == self._points.shape, f"New shape: {new_points.shape}, old shape: {self._points.shape}"
        self._points = new_points

    def create_new(self, lazy=True):
        return self.__class__(self._points, lazy=lazy)

    def _create_kdtree(self):
        return KDTree(self._points)

    @property
    def _kdt(self):
        if self._kdtt is None:
            self._kdtt = self._create_kdtree()
        return self._kdtt

    def _query_kdtree(self, point, **kwargs):
        return self._kdt.query(point, **kwargs)

    def _create_interpolators(self):
        return [interp1d(self._parameters, self._points[:, i]) for i in range(self._dims)]

    def create_point_property_interpolator(self, prop):
        return interp1d(self._parameters, prop)

    def _query_single(self, t):
        return np.array([
            f(t) for f in self._interpolators
        ])

    def _query_vectorized(self, t):
        result = np.empty((t.shape[0], self._dims))

        for i, f in enumerate(self._interpolators):
            result[:, i] = f(t)

        return result

    def __call__(self, t):
        if isinstance(t, (float, int)):
            if t > np.max(self._parameters):
                warn(f"{t} exceeds parameters domain")
            return self._query_single(t)
        else:
            return self._query_vectorized(t)

    def adjacent_points(self, i, j):
        return self._points[i, :], self._points[i, :] + self.difference_vector(i, j)

    def difference_vector(self, i, j):
        return self._points[j, :] - self._points[i, :]

    def unit_difference_vector(self, i, j):
        d = self.difference_vector(i, j)
        return d / np.linalg.norm(d)

    def query(self, t):
        return self(t)

    def distance(self, point):
        _, on_line = self.retract(point)
        return np.linalg.norm(on_line - point)

    def distance_to_other_point_chain(self, other_points):
        assert other_points.shape == self._points.shape
        return np.linalg.norm(other_points - self._points)

    def compute_riemannian_length(self, metric=None, res=1e-3):
        N = self._points.shape[0]
        lens = np.zeros(N - 1)
        for i, j in zip(range(0, N), range(1, N)):
            lens[i] = self.riemannian_length_between_indices(metric, i, j, res=res)

        for failed_idx in np.argwhere(np.isnan(lens)):
            lens[failed_idx] = self.riemannian_length_between_indices(None, failed_idx, failed_idx + 1)

        return np.sum(lens)

    def riemannian_length_between_indices(self, metric, i, j, res=1e-3):
        return compute_length(*self.adjacent_points(i, j), metric, resolution=res)

    def vector_to_curve(self, point):
        _, point_on_curve = self.retract(point)
        return point_on_curve - point

    def derivative(self, t, dt=1e-3, order=5):
        mi, ma = self.range
        ho = order >> 1
        if t - ho * dt < mi:
            tprime = t + ho * dt
        elif t + ho * dt > ma:
            tprime = t - ho * dt
        else:
            tprime = t
        return scipy.misc.derivative(self.__call__, tprime, dx=dt)

    def tangent(self, t):
        d = self.derivative(t)
        return d / np.linalg.norm(d)

    def normal(self, t):
        return np.array([[0, -1], [1, 0]]) @ self.tangent(t)

    def parametrize(self, point):
        t, _ = self.retract(point)
        return t

    def _interpolate_parameters(self, values, weights):
        return np.dot(values, weights) / np.sum(weights)

    def retract(self, point, k=2, func=lambda x: x):
        p = func(self._parameters)
        if len(point.shape) == 1:
            d, i = self._query_kdtree(point, k=k)
            w = np.reciprocal(d)
            if np.isinf(w[0]):
                return p[i[0]], self[i[0]]
            else:
                t = self._interpolate_parameters(p[i], w)
                p = np.einsum("ki,k->i", self._points[i], w) / np.sum(w)
                return t, p
        elif len(point.shape) == 2 and point.shape[1] == self._dims:
            d, i = self._query_kdtree(point, k=k)
            w = np.reciprocal(d)
            acc = np.zeros(point.shape[0])
            exact = np.zeros(point.shape[0])
            points = np.zeros_like(point)
            exact_points = np.zeros_like(point)
            for k in range(d.shape[1]):
                acc += p[i[:, k]] * w[:, k]
                points += self._points[i[k], :] * w[:, k]
                infs = np.isinf(w[:, k])
                exact[infs] = p[i[infs, k]]
                exact_points[infs] = self[i[infs, k]]
            acc /= np.sum(w, axis=1)
            points /= np.sum(w, axis=1)

            exact_matches = np.isinf(np.sum(w, axis=1))
            acc[exact_matches] = exact[exact_matches]
            points[exact_matches] = exact_points[exact_matches]

            return acc, points

    @property
    def arclength(self):
        return self._arclength[-1]

    @property
    def parameters(self):
        return self._parameters

    @property
    def range(self):
        return self._parameters[0], self._parameters[-1]

    def resample(self, n):
        sample_points = np.linspace(0, self._parameters[-1], n)
        return self.query(sample_points)

    def density_aware_resample(self, n):
        return _Curve(
            self.points,
            parameters=np.linspace(0, 1, self.n_points)
        ).resample(n)

    @property
    def shape(self):
        return self.points.shape

    @property
    def points(self):
        return self._points

    def __getitem__(self, item):
        return self._points[item, :]

    @property
    def n_points(self):
        return self.points.shape[0]

    def __iter__(self):
        return iter(self.points)

    @property
    def dimensions(self):
        return self._dims

    def plot(self, ax, *args, **kwargs):
        return ax.plot(self._points[:, 0], self._points[:, 1], *args, **kwargs)

    @property
    def parameter_range(self):
        return np.min(self.parameters), np.max(self.parameters)


class ParametricCurve(_Curve):
    def __init__(self, line, **kwargs):
        al = arclength_parametrization(line)
        super().__init__(line, al / al[-1], **kwargs)


class ArcLengthParametrizedCurve(_Curve):
    def __init__(self, line, **kwargs):
        al = arclength_parametrization(line)
        super().__init__(line, al, **kwargs)


class ClosedCurve(_Curve):
    def __init__(self, points, **kwargs):
        p = np.zeros((points.shape[0] + 1, points.shape[1]))
        p[0:-1, :] = points
        p[-1, :] = points[0, :]
        self._cyclic_points = p
        al = arclength_parametrization(p)
        al = 2*np.pi*al/al[-1] - np.pi
        super().__init__(points, al[0:-1], **kwargs)

    def _interpolate_parameters(self, values, weights):
        sins, coss = np.sin(values), np.cos(values)
        s = np.dot(sins, weights) / np.sum(weights)
        c = np.dot(coss, weights) / np.sum(weights)
        return np.arctan2(s, c)

    def create_point_property_interpolator(self, prop):
        augmented_params = np.zeros(self.n_points + 2)
        augmented_params[1:-1] = self.parameters
        augmented_params[0] = -np.pi if self.parameters[0] > -np.pi + 1e-3 else -np.pi - np.mean(np.diff(self.parameters))
        augmented_params[-1] = np.pi if self.parameters[-1] < np.pi - 1e-3 else np.pi + np.mean(np.diff(self.parameters))
        p = np.r_[prop[-1], prop, prop[0]]
        int = interp1d(augmented_params, p, fill_value='extrapolate')
        
        def f(t):
            new_t = np.arctan2(np.sin(t), np.cos(t))
            return int(new_t)

        return f

    def _create_interpolators(self):
        return [self.create_point_property_interpolator(self.points[:, i]) for i in range(self._dims)]

    def resample(self, n):
        sample_points = np.linspace(0, 2*np.pi, n, endpoint=False)
        return self.query(sample_points)

    def plot(self, ax, *args, **kwargs):
        return ax.plot(self._cyclic_points[:, 0], self._cyclic_points[:, 1], *args, **kwargs)

    @property
    def range(self):
        return -np.pi, np.pi

    @property
    def parameter_range(self):
        return -np.pi, np.pi


class Curve(_Curve):
    def __init__(self, line, params=None, **kwargs):
        if params is None:
            params = arclength_parametrization(line)
        super().__init__(line, params, **kwargs)


def discrete_zero_crossing(x, y, target=0.0):
    assert np.shape(x) == np.shape(y)
    assert np.shape(x)[0] >= 3
    crossings, = np.where(np.diff(np.signbit(y - target)))

    def index_to_x_value(i):
        if i < y.size - 2:
            y_sel = y[i:i + 2]
            x_sel = x[i:i + 2]
            direction = 1 if y_sel[0] < y_sel[1] else -1
            x_target = np.interp(target, y_sel[::direction], x_sel[::direction])

            return x_target
        else:
            return x[i]

    return [*map(index_to_x_value, crossings)]
