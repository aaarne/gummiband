import numpy as np
from links_and_joints.controllers import ImpedanceController
from .lift import lift_to_state_space, compute_velocities


def wrap(q):
    return np.arctan2(np.sin(q), np.cos(q))


class VelIpoModalController:
    def __init__(self, curve, velocities, k=1, d=1, pendulum=None):
        self._c = curve
        self._v = velocities
        self._k, self._d = k, d
        self._v_ipo = [curve.create_point_property_interpolator(velocities[:, i]) for i in range(curve.dimensions)]
        self._pen = pendulum

    @staticmethod
    def generate_from_pendulum(curve, pendulum, E_total, direction, k, d):
        return VelIpoModalController(
            curve=curve,
            velocities=compute_velocities(pendulum, curve, E_total, direction=direction),
            k=k,
            d=d,
            pendulum=pendulum,
        )

    def query(self, q):
        """Returns q_d and dq_d"""
        t, p = self._c.retract(wrap(q))
        v = np.array([v(t) for v in self._v_ipo])
        return t, p, v

    def query_for_parameter(self, t):
        p = self._c(t)
        v = np.array([v(t) for v in self._v_ipo])
        if np.any(np.isnan(p) | np.isnan(v)):
            raise ValueError
        return p, v

    def __call__(self, t, q, dq):
        k = self._k
        d = self._d
        _, p, v = self.query(q)
        delta_q = p - wrap(q)
        delta_dq = v - dq
        return k * delta_q + d * delta_dq

    def parameter(self, q):
        t, _, _ = self.query(q)
        return t

    def desired_state(self, q):
        _, q, dq = self.query(q)
        return q, dq


class TaskCoordinateModalController:
    def __init__(self, curve, pendulum, k=1, xi=1):
        self._c = curve
        self._k, self._xi = k, xi
        self._p = pendulum

    def fkin(self, q):
        t, p = self._c.retract(wrap(q))
        return np.linalg.norm(p - wrap(q))

    def jacobian(self, q):
        t, p = self._c.retract(wrap(q))
        delta_x = (p - wrap(q))
        if np.linalg.norm(delta_x) < 1e-9:
            return np.zeros_like(q)
        else:
            return delta_x / np.linalg.norm(delta_x)

    def __call__(self, t, q, dq):
        x = self.fkin(q)
        jac = self.jacobian(q)

        m = self._p.mass_matrix(q)
        mx = 1 / (jac @ np.linalg.inv(m) @ jac)
        d = 0 #2 * np.sqrt(mx * self._k)

        return -jac * (
                self._k * x
                + d * np.dot(jac, dq)
        )


class TaskCoordinateModalController2(ImpedanceController):
    def __init__(self, curve, pendulum, k=1, xi=.7):
        def fkin(q):
            t, p = curve.retract(wrap(q))
            return np.linalg.norm(p - wrap(q))

        def jacobian(q):
            t, p = curve.retract(wrap(q))
            dq = p - wrap(q)
            return (dq / np.linalg.norm(dq)).reshape((1, -1))

        super().__init__(
            fkin_fun=fkin,
            jacobian_fun=jacobian,
            mass_fun=pendulum.mass_matrix,
            K=k,
            Zeta=xi
        )


class StateSpaceCurveController:
    def __init__(self, lifted_curve, k, d):
        self._c = lifted_curve
        self._k = k
        self._d = d

    @staticmethod
    def generate_from_pendulum(curve, pendulum, E_total, direction, closed, k, d):
        return StateSpaceCurveController(
            lifted_curve=lift_to_state_space(pendulum, curve, E_total, direction, closed=closed),
            k=k,
            d=d,
        )

    def query(self, q, dq):
        y = np.hstack((wrap(q), dq))
        t, yd = self._c.retract(y)
        dq, dqd = np.split(yd, 2)
        return t, dq, dqd

    def __call__(self, t, q, dq):
        k = self._k
        d = self._d
        _, p, v = self.query(q, dq)
        delta_q = p - wrap(q)
        delta_dq = v - dq
        return k*delta_q + d*delta_dq




