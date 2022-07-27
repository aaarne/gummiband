import numpy as np
from gummiband.curves import _Curve, ClosedCurve
from scipy.interpolate import interp1d
import scipy.misc


def dist_on_torus(p1, p2):
    d = p2 - p1
    s, c = np.sin(d), np.cos(d)
    d1, d2 = np.arctan2(s[0], c[0]), np.arctan2(s[1], c[1])
    return np.sqrt(d1 ** 2 + d2 ** 2)


def toroidal_arclength(points):
    lens = np.zeros(points.shape[0] - 1)
    for i in range(points.shape[0] - 1):
        lens[i] = dist_on_torus(points[i], points[i + 1])

    arclength = np.zeros(points.shape[0])
    np.cumsum(lens, out=arclength[1:])
    return arclength


def toroidal_distance(p1, p2):
    d = p1 - p2
    s, c = np.sin(d), np.cos(d)
    return np.arctan2(s, c)


class ToroidalCurve(_Curve):
    def __init__(self, points, parameters=None, **kwargs):
        self.__sins = np.sin(points)
        self.__coss = np.cos(points)
        self._points_big = np.c_[self.__sins, self.__coss]
        points_new = np.arctan2(self.__sins, self.__coss)
        ta = toroidal_arclength(points)
        self._al = ta[-1]
        super().__init__(points_new,
                         parameters=ta if parameters is None else parameters,
                         **kwargs)

    @classmethod
    def create_2d_toroidal_topology(cls, m, n, n_points, noise=False, offset=None):
        points = np.zeros((n_points, 2))
        points[:, 0] = np.linspace(0, 2 * m * np.pi, n_points)
        points[:, 1] = np.linspace(0, 2 * n * np.pi, n_points)
        points = np.arctan2(np.sin(points), np.cos(points))
        if noise:
            points += np.random.normal(0, 0.01, size=points.shape)
        if offset is not None:
            points += offset

        return cls(np.arctan2(np.sin(points), np.cos(points)))

    def derivative(self, t, dt=1e-3, order=5):
        mi, ma = self.range
        ho = order >> 1
        if t - ho * dt < mi:
            tprime = t + ho * dt
        elif t + ho * dt > ma:
            tprime = t - ho * dt
        else:
            tprime = t

        def f(x):
            return np.array([f(x) for f in self._interpolators]).reshape((self._dims, 2)).T

        toroidal_derivative = scipy.misc.derivative(f, tprime, dx=dt)
        pos = f(t)
        return -pos[:, 1] * toroidal_derivative[:, 0] + pos[:, 0] * toroidal_derivative[:, 1]

    def difference_vector(self, i, j):
        d = self._points[j, :] - self._points[i, :]
        return np.arctan2(np.sin(d), np.cos(d))

    def _create_interpolators(self):
        return [interp1d(self._parameters, self._points_big[:, i]) for i in range(2 * self._dims)]

    def _query_single(self, t):
        res = np.array([f(t) for f in self._interpolators]).reshape((self._dims, 2)).T
        return np.arctan2(res[:, 0], res[:, 1])

    def _query_vectorized(self, t):
        res = np.empty((t.shape[0], 2 * self._dims))
        for i, f in enumerate(self._interpolators):
            res[:, i] = f(t)

        res = res.reshape((-1, self._dims, 2)).transpose(0, 2, 1)
        return np.arctan2(res[:, :, 0], res[:, :, 1])

    def distance_to_other_point_chain(self, other_points):
        other_points_big = np.c_[np.sin(other_points), np.cos(other_points)]
        return np.linalg.norm(other_points_big - self._points_big)

    @property
    def arclength(self):
        return self._al


class ClosedToroidalCurve(ToroidalCurve):
    def __init__(self, points, **kwargs):
        p = np.zeros((points.shape[0] + 1, points.shape[1]))
        p[0:-1, :] = points
        p[-1, :] = points[0, :]
        self._cyclic_points = p
        ta = toroidal_arclength(p)
        pars = 2*np.pi*ta/ta[-1] - np.pi
        super().__init__(points, parameters=pars[0:-1], **kwargs)

    def _create_interpolators(self):
        return [self.create_point_property_interpolator(self._points_big[:, i]) for i in range(2*self._dims)]

    def density_aware_resample(self, n):
        pars = 2*np.pi*np.linspace(0, 1, self.n_points) - np.pi
        return ToroidalCurve(self.points, pars)(np.linspace(-np.pi, np.pi, n, endpoint=False))

    _interpolate_parameters = ClosedCurve._interpolate_parameters
    create_point_property_interpolator = ClosedCurve.create_point_property_interpolator
    resample = ClosedCurve.resample
    plot = ClosedCurve.plot
