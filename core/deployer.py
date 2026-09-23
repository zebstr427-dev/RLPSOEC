from dataclasses import dataclass

import numpy as np

from .config import RELAY_BOUNDS, RELAY_INITIAL, RELAY_MAX_SPEED_XY, RELAY_MAX_SPEED_Z


@dataclass
class RelayDeployer:
    position: np.ndarray
    target: np.ndarray
    max_speed_xy: float = RELAY_MAX_SPEED_XY
    max_speed_z: float = RELAY_MAX_SPEED_Z

    @classmethod
    def create(cls):
        initial = np.asarray(RELAY_INITIAL, dtype=float)
        return cls(position=initial.copy(), target=initial.copy())

    def set_target(self, target):
        low = np.asarray(RELAY_BOUNDS[0], dtype=float)
        high = np.asarray(RELAY_BOUNDS[1], dtype=float)
        self.target = np.clip(np.asarray(target, dtype=float), low, high)

    def step(self, seconds: float = 1.0) -> float:
        before = self.position.copy()
        delta = self.target - self.position
        horizontal = float(np.linalg.norm(delta[:2]))
        horizontal_limit = self.max_speed_xy * seconds
        if horizontal > horizontal_limit > 0.0:
            delta[:2] *= horizontal_limit / horizontal
        delta[2] = float(np.clip(delta[2], -self.max_speed_z * seconds, self.max_speed_z * seconds))
        self.position = self.position + delta
        return float(np.linalg.norm(self.position - before))
