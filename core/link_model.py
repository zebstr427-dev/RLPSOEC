from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import BANDWIDTH_HZ, FOLIAGE_LOSS_DB_PER_M, LINK_A, LINK_N
from .dsm import DSMScene


@dataclass(frozen=True)
class LinkEstimate:
    capacity_mbps: float
    snr_db: float
    distance_m: float
    blocked_length_m: float


@dataclass(frozen=True)
class FittedLinkModel:
    bandwidth_hz: float = BANDWIDTH_HZ
    a: float = LINK_A
    n: float = LINK_N
    foliage_loss_db_per_m: float = FOLIAGE_LOSS_DB_PER_M

    @property
    def bandwidth_mbps(self) -> float:
        return self.bandwidth_hz / 1e6

    @staticmethod
    def distance(start: Sequence[float], end: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(start, dtype=float) - np.asarray(end, dtype=float)))

    def los_capacity_mbps(self, distance_m: float) -> float:
        distance = max(float(distance_m), 1e-6)
        snr_linear = self.a * distance ** (-self.n)
        return float(self.bandwidth_mbps * np.log2(1.0 + snr_linear))

    def capacity_mbps(self, distance_m: float, blocked_length_m: float = 0.0) -> float:
        distance = max(float(distance_m), 1e-6)
        loss = 10.0 ** (-self.foliage_loss_db_per_m * max(float(blocked_length_m), 0.0) / 10.0)
        snr_linear = self.a * distance ** (-self.n) * loss
        return float(self.bandwidth_mbps * np.log2(1.0 + snr_linear))

    def estimate(self, start: Sequence[float], end: Sequence[float], scene: Optional[DSMScene]) -> LinkEstimate:
        distance = self.distance(start, end)
        blocked = scene.blocked_length(start, end) if scene is not None else 0.0
        capacity = self.capacity_mbps(distance, blocked)
        snr_linear = max(2.0 ** (capacity / self.bandwidth_mbps) - 1.0, 1e-12)
        return LinkEstimate(
            capacity_mbps=capacity,
            snr_db=float(10.0 * np.log10(snr_linear)),
            distance_m=distance,
            blocked_length_m=blocked,
        )

    def fit(self, distances: Sequence[float], capacities_mbps: Sequence[float]):
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise RuntimeError("scipy is required for link fitting") from exc
        distances = np.asarray(distances, dtype=float)
        capacities_mbps = np.asarray(capacities_mbps, dtype=float)
        if distances.shape != capacities_mbps.shape or distances.ndim != 1:
            raise ValueError("distances and capacities must be one-dimensional and aligned")
        if np.any(distances <= 0.0) or np.any(capacities_mbps < 0.0):
            raise ValueError("distances must be positive and capacities non-negative")

        def residual(parameters):
            a, n = parameters
            predicted = self.bandwidth_mbps * np.log2(1.0 + a * distances ** (-n))
            return predicted - capacities_mbps

        result = least_squares(residual, (self.a, self.n), bounds=((1e-9, 1e-9), (np.inf, np.inf)))
        return FittedLinkModel(self.bandwidth_hz, float(result.x[0]), float(result.x[1]), self.foliage_loss_db_per_m)
