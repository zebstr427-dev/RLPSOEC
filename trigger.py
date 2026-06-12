from __future__ import annotations

import numpy as np


class Trigger:
    """Communication-aware on-demand redeployment trigger for RLPSOEC."""

    def __init__(
        self,
        avg_snr_thresh=0.9,
        snr_fluct_thresh=2.0,
        avg_cap_thresh=0.9,
        cap_fluct_thresh=1.0,
        time_interval=50,
        history_len=1000,
        verbose=False,
    ):
        self.avg_snr_thresh = avg_snr_thresh
        self.snr_fluct_thresh = snr_fluct_thresh
        self.avg_cap_thresh = avg_cap_thresh
        self.cap_fluct_thresh = cap_fluct_thresh
        self.time_interval = time_interval
        self.history_len = history_len
        self.verbose = verbose

        self.snr_history = []
        self.cap_history = []
        self.last_trigger_time = -time_interval

    def _push(self, history, value):
        history.append(value)
        if len(history) > self.history_len:
            history.pop(0)

    def should_trigger(self, current_snr, current_cap, current_time):
        self._push(self.snr_history, current_snr)
        self._push(self.cap_history, current_cap)

        trigger_avg_snr = (
            len(self.snr_history) > 1
            and current_snr < np.mean(self.snr_history[:-1]) * self.avg_snr_thresh
        )
        trigger_avg_cap = (
            len(self.cap_history) > 1
            and current_cap < np.mean(self.cap_history[:-1]) * self.avg_cap_thresh
        )
        trigger_snr_fluct = (
            len(self.snr_history) > 1
            and abs(current_snr - self.snr_history[-2]) > self.snr_fluct_thresh
        )
        trigger_cap_fluct = (
            len(self.cap_history) > 1
            and abs(current_cap - self.cap_history[-2]) > self.cap_fluct_thresh
        )
        trigger_periodic = (current_time - self.last_trigger_time) >= self.time_interval

        if self.verbose:
            print(
                f"[Trigger t={current_time}] "
                f"avg_snr_drop={trigger_avg_snr}, "
                f"snr_fluct={trigger_snr_fluct}, "
                f"avg_cap_drop={trigger_avg_cap}, "
                f"cap_fluct={trigger_cap_fluct}, "
                f"periodic={trigger_periodic}"
            )

        if trigger_avg_snr or trigger_snr_fluct or trigger_avg_cap or trigger_cap_fluct or trigger_periodic:
            self.last_trigger_time = current_time
            return True
        return False
