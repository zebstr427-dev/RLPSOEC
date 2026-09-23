from collections import deque

import numpy as np

from .config import (
    TRIGGER_AVERAGE_FACTOR,
    TRIGGER_CAPACITY_JUMP_MBPS,
    TRIGGER_PERIOD_STEPS,
    TRIGGER_SNR_JUMP_DB,
)


class CommunicationTrigger:
    def __init__(self, period_steps=TRIGGER_PERIOD_STEPS, history_size=20):
        self.period_steps = int(period_steps)
        self.capacity = deque(maxlen=history_size)
        self.snr = deque(maxlen=history_size)
        self.last_trigger = -self.period_steps

    def check(self, step: int, total_capacity: float, average_snr: float):
        if not self.capacity:
            periodic = True
            average_drop = False
            snr_drop = False
            capacity_jump = False
            snr_jump = False
        else:
            capacity_mean = float(np.mean(self.capacity))
            snr_mean = float(np.mean(self.snr))
            average_drop = total_capacity < TRIGGER_AVERAGE_FACTOR * capacity_mean
            snr_drop = average_snr < TRIGGER_AVERAGE_FACTOR * snr_mean
            capacity_jump = abs(total_capacity - self.capacity[-1]) > TRIGGER_CAPACITY_JUMP_MBPS
            snr_jump = abs(average_snr - self.snr[-1]) > TRIGGER_SNR_JUMP_DB
            periodic = step - self.last_trigger >= self.period_steps
        reasons = {
            "average_capacity": bool(average_drop),
            "average_snr": bool(snr_drop),
            "capacity_fluctuation": bool(capacity_jump),
            "snr_fluctuation": bool(snr_jump),
            "periodic": bool(periodic),
        }
        return bool(any(reasons.values())), reasons

    def observe(self, step: int, total_capacity: float, average_snr: float, triggered: bool):
        self.capacity.append(float(total_capacity))
        self.snr.append(float(average_snr))
        if triggered:
            self.last_trigger = int(step)


def fixed_trigger_schedule(steps: int, rate: float = 0.769):
    count = max(1, int(round(steps * rate)))
    return set(np.rint(np.linspace(0, steps - 1, count)).astype(int).tolist())
