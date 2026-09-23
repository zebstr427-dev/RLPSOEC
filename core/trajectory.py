from typing import Tuple

import numpy as np

from .config import MISSION_STEPS, SURVEY_ALTITUDE, SURVEY_PASSES, SURVEY_SPEED


def z_scan_trajectory(
    x_start: float,
    x_end: float,
    steps: int = MISSION_STEPS,
    passes: int = SURVEY_PASSES,
    speed_mps: float = SURVEY_SPEED,
    altitude_m: float = SURVEY_ALTITUDE,
    y_start: float = 100.0,
    y_spacing: float = 200.0,
) -> np.ndarray:
    if steps % passes:
        raise ValueError("mission steps must be divisible by the number of passes")
    per_pass = steps // passes
    sweep = abs(float(x_end) - float(x_start))
    points = []
    for pass_index in range(passes):
        y = y_start + pass_index * y_spacing
        direction = 1.0 if pass_index % 2 == 0 else -1.0
        for sample in range(per_pass):
            distance = min(sample * speed_mps, sweep)
            x = x_start + distance if direction > 0 else x_end - distance
            points.append((x, y, altitude_m))
    return np.asarray(points, dtype=float)


def paired_trajectories(steps: int = MISSION_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    first = z_scan_trajectory(0.0, 500.0, steps=steps)
    second = z_scan_trajectory(500.0, 1000.0, steps=steps, y_start=150.0, y_spacing=170.0)
    return first, second
