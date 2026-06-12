from __future__ import annotations

from collections import deque

import numpy as np


class RewardTracker:
    """Normalizes communication gain and convergence speed into PPO rewards."""

    def __init__(self, history_len: int = 100):
        self.comm_imp_hist = deque(maxlen=history_len)
        self.conv_speed_hist = deque(maxlen=history_len)

    def compute(self, comm_imp: float, convergence_speed: float) -> float:
        self.comm_imp_hist.append(comm_imp)
        self.conv_speed_hist.append(convergence_speed)
        mean_ci = np.mean(self.comm_imp_hist) if self.comm_imp_hist else 1.0
        std_ci = np.std(self.comm_imp_hist) if self.comm_imp_hist else 1.0
        mean_cv = np.mean(self.conv_speed_hist) if self.conv_speed_hist else 1.0
        std_cv = np.std(self.conv_speed_hist) if self.conv_speed_hist else 1.0
        norm_ci = (comm_imp - mean_ci) / (std_ci + 1e-6)
        norm_cv = (convergence_speed - mean_cv) / (std_cv + 1e-6)
        return float(np.tanh(norm_ci + 0.5 * norm_cv) * 5.0)
