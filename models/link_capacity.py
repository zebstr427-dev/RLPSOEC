from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np


@dataclass(frozen=True)
class MeasurementDrivenLinkModel:
    """Measurement-driven effective-capacity model used by RLPSOEC.

    The model treats field throughput as effective capacity. LOS links are
    estimated by a fitted capacity-distance curve. NLoS links reuse the same
    curve after applying an additional foliage/terrain penetration loss.
    """

    bandwidth_hz: float = 10e6
    gain: float = 8.0e8
    path_loss_exponent: float = 2.35
    min_distance_m: float = 1.0
    max_capacity_mbps: float = 260.0
    capacity_scale: float = 1.0
    foliage_loss_db_per_m: float = 0.03
    equivalent_snr_offset_db: float = 0.0

    @classmethod
    def default(cls, **overrides) -> "MeasurementDrivenLinkModel":
        params = {
            "bandwidth_hz": 10e6,
            "gain": 8.0e8,
            "path_loss_exponent": 2.35,
            "min_distance_m": 1.0,
            "max_capacity_mbps": 260.0,
            "capacity_scale": 1.0,
            "foliage_loss_db_per_m": 0.03,
            "equivalent_snr_offset_db": 0.0,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_report(cls, report_path: str | Path) -> "MeasurementDrivenLinkModel":
        path = Path(report_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        fit = data.get("fit", {})
        settings = data.get("link_model_settings", {})
        capacity_scale = float(settings.get("capacity_scale", fit.get("capacity_scale", 1.0)))
        return cls.default(
            bandwidth_hz=float(fit.get("bandwidth_hz", 10e6)),
            gain=float(fit.get("gain", 8.0e8)),
            path_loss_exponent=float(fit.get("path_loss_exponent", 2.35)),
            max_capacity_mbps=float(fit.get("max_capacity_mbps", 260.0)),
            capacity_scale=capacity_scale,
            foliage_loss_db_per_m=float(fit.get("foliage_loss_db_per_m", 0.03)),
            equivalent_snr_offset_db=float(settings.get("equivalent_snr_offset_db", 0.0)),
        )

    def capacity_for_distance(self, distance_m: float, blockage_distance: float = 0.0) -> float:
        distance = max(float(distance_m), self.min_distance_m)
        blockage = max(float(blockage_distance), 0.0)

        snr_like = self.gain / (distance ** self.path_loss_exponent)
        if blockage > 0.0:
            loss_linear = 10.0 ** (-(self.foliage_loss_db_per_m * blockage) / 10.0)
            snr_like *= loss_linear

        bandwidth_mbps = self.bandwidth_hz / 1e6
        capacity = bandwidth_mbps * math.log2(1.0 + max(snr_like, 0.0))
        capacity *= self.capacity_scale
        return float(np.clip(capacity, 0.0, self.max_capacity_mbps))

    def equivalent_snr_db(self, capacity_mbps: float) -> float:
        bandwidth_mbps = max(self.bandwidth_hz / 1e6, 1e-9)
        clipped_capacity = max(float(capacity_mbps), 0.0)
        snr_linear = max(2.0 ** (clipped_capacity / bandwidth_mbps) - 1.0, 1e-12)
        return float(10.0 * math.log10(snr_linear) + self.equivalent_snr_offset_db)

    def equivalent_snr_linear(self, capacity_mbps: float) -> float:
        return float(10.0 ** (self.equivalent_snr_db(capacity_mbps) / 10.0))

    def capacity_from_equivalent_snr_db(self, snr_db: float) -> float:
        bandwidth_mbps = max(self.bandwidth_hz / 1e6, 1e-9)
        adjusted = float(snr_db) - self.equivalent_snr_offset_db
        snr_linear = 10.0 ** (adjusted / 10.0)
        return float(bandwidth_mbps * math.log2(1.0 + snr_linear))
