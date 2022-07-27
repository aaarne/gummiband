import numpy as np
from .curves import Curve, ClosedCurve


def compute_velocities(pendulum, curve, E_total, direction=1):
    vels = np.zeros_like(curve.points)
    for i, p in enumerate(curve):
        tangent = curve.tangent(curve.parameters[i])
        U = pendulum.potential_energy(p)
        alpha = np.sqrt(2 * (E_total - U) / np.einsum("ij,i,j", pendulum.M(p), tangent, tangent))
        vel = alpha * tangent * direction
        vels[i, :] = vel
    return vels


def plot_lifted_curve(ax, curve, style='bo-', closed=False):
    n = curve.dimensions
    print(n / 2)
    points = curve.points[:, 0:n // 2]
    vels = curve.points[:, n // 2:]

    ax.plot(points[:, 0], points[:, 1], style)
    if closed:
        ax.plot(points[[-1, 0], 0], points[[-1, 0], 1], style)

    ax.quiver(points[:, 0], points[:, 1], vels[:, 0], vels[:, 1], angles='xy')


def lift_to_state_space(pendulum, curve, E_total, direction=1, closed=False):
    lifted_points = np.c_[curve.points, compute_velocities(pendulum, curve, E_total, direction=direction)]
    if closed:
        return Curve(lifted_points)
    else:
        return ClosedCurve(lifted_points)
