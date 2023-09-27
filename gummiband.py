import warnings

import numpy as np
from allerlei import progressify, Timer
from .metric import Euclidean


class CollapsedToPointException(Exception):
    """The Gummiband collapsed too far."""
    pass


class Gummiband:
    def __init__(self, initial_curve, metric=None, res=1e-2, fix_endpoints=False, history=True):
        self._res = res
        self._gb = initial_curve
        l = self._gb.compute_riemannian_length(metric, res=res)
        self._points_per_length = initial_curve.n_points / initial_curve.arclength
        self._lens = [l]
        self._metric = metric if metric else Euclidean
        self._fix_enpoints = fix_endpoints
        self._history = history
        self._chr_forces = None
        if history:
            self._h = [initial_curve.points.copy()]

    def adjacent_point_iterator(self):
        for i, _ in enumerate(self._gb):
            pred = (i - 1) % self._gb.n_points
            succ = (i + 1) % self._gb.n_points
            yield pred, i, succ

    def compute_forces(self):
        f = self.compute_lin_forces()
        if not self._metric.is_euclidean:
            f += .25 * self.compute_metric_forces()

        if self._fix_enpoints:
            f[[0, -1], :] = 0

        return f

    def compute_metric_forces(self):
        metric_forces = np.zeros_like(self._gb.points)

        for pred, i, succ in self.adjacent_point_iterator():
            gamma = self._metric.christoffel_second_kind(self._gb[i])
            hop = self._gb.difference_vector(succ, pred)
            metric_forces[i] = np.einsum("kij,i,j->k", gamma, hop, hop)

        return metric_forces

    def compute_lin_forces(self):
        metric, gb = self._metric, self._gb
        forces = np.zeros_like(gb.points)

        for pred, i, succ in self.adjacent_point_iterator():
            forces[i] = gb.difference_vector(i, pred) + gb.difference_vector(i, succ)

        return forces

    def plot_with_forces(self, ax):
        self._gb.plot(ax, 'o-')
        f = self.compute_forces()
        p = self._gb.points
        ax.quiver(p[:, 0], p[:, 1], f[:, 0], f[:, 1], angles='xy')

    def optimize(self,
                 epochs=None,
                 steps=100,
                 lr=1e-2,
                 verbose=True,
                 force_resampling=False,
                 schedule=None,
                 n_points=None,
                 caching=False):
        epoch = 0
        v = np.inf
        last_com = np.mean(self.curve.points, axis=0)
        com_vel = np.inf
        total_steps = 0
        while True:
            if schedule:
                try:
                    res = schedule(epoch, self.loss(), v, self.curve, com_vel)
                    if res:
                        l, s, n = res
                        lr = l if l else lr
                        steps = s if s else steps
                        n_points = n if n else n_points
                except StopIteration:
                    break
            if verbose:
                print(f"--- Epoch {epoch + 1}{f'/{epochs}' if epochs else ''}")
            try:
                v = self.step(steps,
                              dt=lr,
                              verbose=verbose,
                              force_resampling=force_resampling,
                              target_points=n_points,
                              caching=caching)
                total_steps += steps
                self._lens.append(self.loss())
                com = np.mean(self.curve.points, axis=0)
                com_vel = np.linalg.norm(com - last_com)
                last_com = com
            except CollapsedToPointException:
                print("The Gummiband collapsed.")
                break

            epoch += 1
            yield self.curve, self.loss(), v, com_vel, total_steps

            if epochs:
                if epoch >= epochs:
                    break

    def step(self,
             steps=1,
             dt=0.1,
             force_resampling=False,
             verbose=False,
             target_points=None,
             caching=False):
        if target_points:
            if self._gb.n_points != target_points:
                force_resampling = True

        if force_resampling:
            if target_points:
                n_target = target_points
            else:
                n_target = int(np.ceil(self._points_per_length * self._gb.arclength))
            if n_target > 1000000:
                print("Refuse to resample more than a million points")
                raise ValueError
            if verbose:
                print(f"Resample to {n_target} points.")
            self._gb = self._gb.__class__(self._gb.density_aware_resample(n_target), lazy=True)
        points_before = self._gb.points.copy()
        metric_forces = self.compute_metric_forces() if caching else np.zeros_like(points_before)
        ds = self._gb.arclength / self.curve.n_points
        rate = dt/(ds**2)

        def get_metric_forces():
            if caching:
                return metric_forces
            else:
                return self.compute_metric_forces()

        for _ in progressify(range(steps), show=verbose):
            forces = self.compute_lin_forces()

            if not self._metric.is_euclidean:
                forces += .25 * get_metric_forces()

            if self._fix_enpoints:
                forces[[0, -1], :] = 0

            points = self._gb.points + rate * forces
            self._gb.override_points(points)
        else:
            self._gb = self._gb.create_new()

        if self._history:
            self._h.append(self._gb.points.copy())

        v = self._gb.distance_to_other_point_chain(points_before)

        if self._gb.n_points < 2:
            raise CollapsedToPointException

        return v

    def loss(self):
        return self._gb.compute_riemannian_length(self._metric, res=self._res)

    @property
    def curve(self):
        return self._gb

    @property
    def length_history(self):
        return self._lens

    @property
    def history(self):
        if self._history:
            try:
                return np.array(self._h)
            except ValueError:
                warnings.warn("History has no single shape. Returning empty array.")
                return np.array([])
        else:
            raise ValueError("History not actived")

    def is_valid(self):
        return all(self._metric.valid(q) for q in self._gb)