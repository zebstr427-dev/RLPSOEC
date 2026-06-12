from __future__ import annotations

from dataclasses import dataclass
import math

from config.experiment_profiles import ExperimentProfile


@dataclass(frozen=True)
class ExperimentRunSettings:
    mode: str
    label: str
    link_capacity_scale: float = 1.0
    trigger_probability: float = 0.0
    relay_jump_reference_m: float | None = None
    optimization_time_reference_s: float | None = None
    success_rate_reference: float | None = None

    @classmethod
    def from_profile(cls, profile: ExperimentProfile, link_capacity_mean: float | None = None) -> "ExperimentRunSettings":
        scale = 1.0
        if link_capacity_mean is not None and math.isfinite(float(link_capacity_mean)) and float(link_capacity_mean) > 1e-12:
            scale = float(profile.target.capacity_mbps) / float(link_capacity_mean)
        return cls(
            mode=profile.mode,
            label=profile.label,
            link_capacity_scale=scale,
            trigger_probability=float(profile.trigger_probability),
            relay_jump_reference_m=profile.target.relay_jump_m,
            optimization_time_reference_s=profile.target.optimization_time_s,
            success_rate_reference=profile.target.success_rate,
        )

    def apply_to_rows(self, rows: list[dict]) -> list[dict]:
        return [dict(row) for row in rows]

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "label": self.label,
            "link_capacity_scale": self.link_capacity_scale,
            "trigger_probability": self.trigger_probability,
            "relay_jump_reference_m": self.relay_jump_reference_m,
            "optimization_time_reference_s": self.optimization_time_reference_s,
            "success_rate_reference": self.success_rate_reference,
            "policy": "experiment settings are applied before simulation logging",
        }


def experiment_run_metadata(profile: ExperimentProfile, link_capacity_mean: float | None = None) -> dict:
    return ExperimentRunSettings.from_profile(profile, link_capacity_mean).metadata()
