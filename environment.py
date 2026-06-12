from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from models.link_capacity import MeasurementDrivenLinkModel


class Environment:
    """RLPSOEC communication environment.

    This compatibility class keeps the historical get_capacity/get_snr API while
    replacing the old pure log-distance Shannon model with a measurement-driven
    effective-capacity model aligned with the paper.
    """

    def __init__(self, dsm_map, cell_size=1.0, link_model: MeasurementDrivenLinkModel | None = None):
        self.dsm = np.asarray(dsm_map, dtype=float)
        self.cell_size = float(cell_size)
        self.map_h, self.map_w = self.dsm.shape
        self.survey_uavs = []
        self.link_model = link_model or MeasurementDrivenLinkModel.default()

        # Compatibility attributes retained for scripts that inspect them.
        self.Pt_dBm = 20
        self.fc = 2.4e9
        self.B = self.link_model.bandwidth_hz
        self.N0_dBmHz = -174
        self.gamma = self.link_model.path_loss_exponent
        self.LOS_penalty_dB = 0
        self.NLOS_penalty_dB = 40
        self._blockage_cache = {}
        self._capacity_cache = {}

    @staticmethod
    def _distance(p1, p2) -> float:
        return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))

    def _line_samples(self, p1, p2, num_points=80):
        p1_arr = np.asarray(p1, dtype=float)
        p2_arr = np.asarray(p2, dtype=float)
        for alpha in np.linspace(0.0, 1.0, num_points):
            yield p1_arr + (p2_arr - p1_arr) * alpha

    def is_los(self, p1, p2):
        return self.get_blockage_distance(p1, p2) <= 0.0

    def get_blockage_distance(self, p1, p2, num_points=80):
        key = self._cache_key(p1, p2, "block", num_points)
        if key in self._blockage_cache:
            return self._blockage_cache[key]

        p1_arr = np.asarray(p1, dtype=float)
        p2_arr = np.asarray(p2, dtype=float)
        alphas = np.linspace(0.0, 1.0, num_points)
        samples = p1_arr + (p2_arr - p1_arr) * alphas[:, None]
        if len(samples) < 2:
            return 0.0
        segment_distance = self._distance(p1, p2) / max(len(samples) - 1, 1)
        x_idx = (samples[:, 0] / self.cell_size).astype(int)
        y_idx = (samples[:, 1] / self.cell_size).astype(int)
        valid = (x_idx >= 0) & (x_idx < self.map_w) & (y_idx >= 0) & (y_idx < self.map_h)
        blocked = np.zeros(num_points, dtype=bool)
        blocked[valid] = samples[valid, 2] < self.dsm[y_idx[valid], x_idx[valid]]
        blocked_length = float(np.count_nonzero(blocked) * segment_distance)
        self._blockage_cache[key] = blocked_length
        return blocked_length

    def get_block_ratio(self, p1, p2):
        distance = max(self._distance(p1, p2), 1e-9)
        return float(np.clip(self.get_blockage_distance(p1, p2) / distance, 0.0, 1.0))

    def get_capacity(self, tx_pos, rx_pos):
        key = self._cache_key(tx_pos, rx_pos, "capacity", 0)
        if key in self._capacity_cache:
            return self._capacity_cache[key]
        distance = self._distance(tx_pos, rx_pos)
        blockage = self.get_blockage_distance(tx_pos, rx_pos)
        capacity = self.link_model.capacity_for_distance(distance, blockage)
        self._capacity_cache[key] = capacity
        return capacity

    def get_snr(self, tx_pos, rx_pos):
        capacity = self.get_capacity(tx_pos, rx_pos)
        return self.link_model.equivalent_snr_linear(capacity)

    def get_snr_db(self, tx_pos, rx_pos):
        capacity = self.get_capacity(tx_pos, rx_pos)
        return self.link_model.equivalent_snr_db(capacity)

    @staticmethod
    def _quantized_pos(pos):
        return tuple(np.round(np.asarray(pos, dtype=float), 2))

    def _cache_key(self, p1, p2, kind, extra):
        return (kind, self._quantized_pos(p1), self._quantized_pos(p2), extra)


def test_snr_vs_distance(env, tx, d_min=1, d_max=500, n=100):
    distances = np.linspace(d_min, d_max, n)
    snr_db = [env.get_snr_db(tx, (d, 0, tx[2])) for d in distances]

    print("Distance (m) first:", np.round(distances[:5], 1), "last:", np.round(distances[-5:], 1))
    print("Equivalent SNR (dB) first:", np.round(snr_db[:5], 2), "last:", np.round(snr_db[-5:], 2))

    plt.figure(figsize=(6, 3))
    plt.plot(distances, snr_db, lw=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Equivalent SNR (dB)")
    plt.title("Equivalent SNR vs. Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_capacity_vs_distance(env, tx, d_min=1, d_max=500, n=100):
    distances = np.linspace(d_min, d_max, n)
    caps = [env.get_capacity(tx, (d, 0, tx[2])) for d in distances]

    print("Distance (m) first:", np.round(distances[:5], 1), "last:", np.round(distances[-5:], 1))
    print("Capacity (Mbps) first:", np.round(caps[:5], 2), "last:", np.round(caps[-5:], 2))

    plt.figure(figsize=(6, 3))
    plt.plot(distances, caps, lw=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Capacity (Mbps)")
    plt.title("Measurement-Driven Capacity vs. Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
