from __future__ import annotations

import csv
import math
from pathlib import Path

from config.experiment_profiles import ExperimentProfile
from .schema import canonicalize_rows
from .success import success_rate_from_rows


def finite_values(rows: list[dict[str, float]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, float("nan"))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def mean_or_zero(rows: list[dict[str, float]], key: str) -> float:
    values = finite_values(rows, key)
    return float(sum(values) / len(values)) if values else 0.0


def summarize_rows(rows: list[dict], profile: ExperimentProfile | None = None) -> dict[str, float]:
    canonical = canonicalize_rows(rows)
    return {
        "capacity": mean_or_zero(canonical, "capacity"),
        "snr": mean_or_zero(canonical, "snr"),
        "trigger_rate": mean_or_zero(canonical, "triggered"),
        "relay_jump": mean_or_zero(canonical, "relay_jump"),
        "comm_imp": mean_or_zero(canonical, "comm_imp"),
        "reward": mean_or_zero(canonical, "reward"),
        "opt_time": mean_or_zero(canonical, "optimization_time"),
        "log_success_rate": success_rate_from_rows(canonical),
        "success_rate": success_rate_from_rows(canonical, profile),
        "rows": float(len(canonical)),
    }


def summarize_csv(path: str | Path, profile: ExperimentProfile | None = None) -> dict[str, float]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return summarize_rows(rows, profile)
