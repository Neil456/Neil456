"""Signal smoothing.

Raw hand landmarks jitter by a few pixels every frame. Sending that straight to
the servos makes the arm buzz. The One Euro filter (Casiez et al., CHI 2012) is
the standard fix for interactive tracking: heavy smoothing when the hand is
still (kills jitter), light smoothing when it moves fast (keeps lag low).
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Vectorised One Euro filter.

    min_cutoff: lower = smoother when still (more lag). Hz.
    beta:       higher = less lag during fast motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray:
        tau = 1.0 / (2 * math.pi * np.asarray(cutoff))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self._t, 1e-4)
        self._t = t

        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class RateLimiter:
    """Caps how fast each joint command may change (deg/s). Protects the arm
    from sudden jumps, e.g. when tracking (re)starts or the hand flickers."""

    def __init__(self, max_rate: dict[str, float]):
        self.max_rate = max_rate
        self.state: dict[str, float] | None = None
        self._t: float | None = None

    def reset(self, state: dict[str, float], t: float) -> None:
        self.state = dict(state)
        self._t = t

    def __call__(self, target: dict[str, float], t: float) -> dict[str, float]:
        if self.state is None or self._t is None:
            self.reset(target, t)
            return dict(target)
        dt = min(max(t - self._t, 0.0), 0.1)  # don't allow huge steps after a stall
        self._t = t
        out = {}
        for k, v in target.items():
            prev = self.state.get(k, v)
            step = self.max_rate.get(k, 1e9) * dt
            out[k] = prev + max(-step, min(step, v - prev))
        self.state = out
        return dict(out)
