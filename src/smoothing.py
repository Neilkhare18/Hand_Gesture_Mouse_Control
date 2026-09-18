"""Jitter suppression and motion smoothing filters for low-latency HCI cursor control."""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """Standard first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y_prev: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.y_prev is None:
            self.y_prev = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self.y_prev
        self.y_prev = filtered
        return filtered

    def reset(self) -> None:
        self.y_prev = None


class OneEuroFilter1D:
    """1D implementation of the 1€ Filter (Casiez et al., CHI 2012).
    
    Dynamically adjusts low-pass filter cutoff frequency based on input speed:
    - Low velocity: Low cutoff -> heavy jitter/tremor suppression.
    - High velocity: High cutoff -> low lag/instantaneous response.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None
        self.x_prev: Optional[float] = None

    @staticmethod
    def _compute_alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.033
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            self.x_prev = x
            self.x_filter.reset()
            self.dx_filter.reset()
            return self.x_filter.filter(x, 1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        # Guard against zero or negative dt
        if dt <= 0.0:
            dt = 0.001
        rate = 1.0 / dt

        # Estimate derivative of the signal
        dx = (x - self.x_prev) * rate if self.x_prev is not None else 0.0
        self.x_prev = x

        # Filter the derivative
        alpha_d = self._compute_alpha(rate, self.d_cutoff)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Dynamic cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the signal
        alpha = self._compute_alpha(rate, cutoff)
        return self.x_filter.filter(x, alpha)

    def reset(self) -> None:
        self.last_time = None
        self.x_prev = None
        self.x_filter.reset()
        self.dx_filter.reset()


class OneEuroFilter2D:
    """2D wrapper for cursor (x, y) smoothing using One Euro Filter."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.04, d_cutoff: float = 1.0):
        self.x_filter = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.y_filter = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.perf_counter()
        fx = self.x_filter.filter(x, timestamp)
        fy = self.y_filter.filter(y, timestamp)
        return fx, fy

    def reset(self) -> None:
        self.x_filter.reset()
        self.y_filter.reset()


class AdaptiveEMA2D:
    """Velocity-aware Exponential Moving Average for 2D cursor coordinates."""

    def __init__(self, base_alpha: float = 0.35, min_alpha: float = 0.15, max_alpha: float = 0.90, k: float = 0.02):
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.k = k
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = x
            self.prev_y = y
            return x, y

        dist = math.hypot(x - self.prev_x, y - self.prev_y)
        dynamic_alpha = max(self.min_alpha, min(self.max_alpha, self.base_alpha + self.k * dist))

        fx = dynamic_alpha * x + (1.0 - dynamic_alpha) * self.prev_x
        fy = dynamic_alpha * y + (1.0 - dynamic_alpha) * self.prev_y

        self.prev_x = fx
        self.prev_y = fy
        return fx, fy

    def reset(self) -> None:
        self.prev_x = None
        self.prev_y = None


class DeadbandFilter:
    """Deadband suppressor to ignore sub-threshold micro-jitters."""

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        if self.last_x is None or self.last_y is None:
            self.last_x = x
            self.last_y = y
            return x, y

        dx = x - self.last_x
        dy = y - self.last_y
        dist = math.hypot(dx, dy)

        if dist < self.threshold:
            return self.last_x, self.last_y

        self.last_x = x
        self.last_y = y
        return x, y

    def reset(self) -> None:
        self.last_x = None
        self.last_y = None
