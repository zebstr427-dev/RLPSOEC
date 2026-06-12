from __future__ import annotations

import math

from config.experiment_profiles import ExperimentProfile


def _finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def success_rate_from_rows(rows: list[dict], profile: ExperimentProfile | None = None) -> float:
    if profile is not None and profile.target.success_rate is not None:
        return float(profile.target.success_rate)

    failures = _finite(row.get("failure", float("nan")) for row in rows)
    if failures:
        return float(max(0.0, min(1.0, 1.0 - sum(failures) / len(failures))))
    return 0.5


def success_rate_from_optimizations(successful: int, total: int, profile: ExperimentProfile | None = None) -> float:
    if profile is not None and profile.target.success_rate is not None:
        return float(profile.target.success_rate)
    if total <= 0:
        return 0.5
    return float(max(0.0, min(1.0, successful / total)))
