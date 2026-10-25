"""One-Euro temporal smoothing for 3D joints (Casiez et al., CHI 2012).
Adaptive low-pass: heavy smoothing when a joint is still, light smoothing when it moves fast."""
from __future__ import annotations
import math
import numpy as np


class OneEuroFilter:
    """min_cutoff (Hz): jitter smoothing at rest. beta: speed coefficient (units: Hz per m/s for metric inputs)."""
    def __init__(self, min_cutoff: float = 1.5, beta: float = 5.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t = self.x = self.dx = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.t = self.x = self.dx = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.t is None or t <= self.t:
            self.t, self.x, self.dx = t, x, np.zeros_like(x)
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat)))
        x_hat = a * x + (1 - a) * self.x
        self.t, self.x, self.dx = t, x_hat, dx_hat
        return x_hat


class SkeletonSmoother:
    """Applies one OneEuroFilter per joint to skeleton dicts from postprocess.decode_skeleton.
    A joint that disappears (xyz None) resets its filter so stale positions never leak back in."""
    def __init__(self, n_joints: int = 17, **kw):
        self.f = [OneEuroFilter(**kw) for _ in range(n_joints)]

    def __call__(self, skeleton: dict, t: float) -> dict:
        for j, f in zip(skeleton["joints"], self.f):
            if j["xyz"] is None:
                f.reset()
            else:
                j["xyz"] = [float(v) for v in f(j["xyz"], t)]
        return skeleton
