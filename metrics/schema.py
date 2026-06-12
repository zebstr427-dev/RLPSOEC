from __future__ import annotations

import math


ALIASES: dict[str, tuple[str, ...]] = {
    "step": ("t", "step"),
    "snr": ("avgsnrdb", "avgsnr", "snr"),
    "capacity": ("totalcap", "totalcapacity", "capacity", "throughput"),
    "triggered": ("triggered", "trigger"),
    "comm_imp": ("commimp", "communicationimprovement", "capacitygain"),
    "convergence_speed": ("convergencespeed",),
    "reward": ("reward",),
    "relay_jump": ("relayjump", "jumpdistance", "relaymovement", "movement"),
    "failure": ("failureflag", "failure"),
    "optimization_time": ("optimizationtime", "opttime"),
}


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_column(columns, metric: str) -> str | None:
    keys = ALIASES.get(metric, (metric,))
    normalized = [(normalize_name(col), col) for col in columns]
    for key in keys:
        for norm, original in normalized:
            if key == norm or key in norm:
                return original
    return None


def parse_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def canonicalize_row(row: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    columns = list(row.keys())
    for metric in ALIASES:
        column = find_column(columns, metric)
        out[metric] = parse_float(row.get(column)) if column is not None else float("nan")
    return out


def canonicalize_rows(rows: list[dict]) -> list[dict[str, float]]:
    return [canonicalize_row(row) for row in rows]
