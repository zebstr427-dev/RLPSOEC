from __future__ import annotations

from config.experiment_profiles import PaperTarget, StabilityTolerance, all_profiles


def _pct_diff(actual: float, target: float) -> float:
    if abs(target) < 1e-12:
        return 0.0 if abs(actual) < 1e-12 else float("inf")
    return (actual - target) / target * 100.0


def _metric_check(actual: float, target: float | None, tolerance: float, percent: bool) -> dict:
    if target is None:
        return {"status": "not_applicable", "actual": actual, "target": None, "delta": None}
    delta = _pct_diff(actual, target) if percent else actual - target
    passed = abs(delta) <= tolerance
    near = abs(delta) <= tolerance * 1.8
    status = "passed" if passed else ("review" if near else "failed")
    return {"status": status, "actual": actual, "target": target, "delta": delta, "tolerance": tolerance}


def _combine_status(statuses: list[str]) -> str:
    active = [s for s in statuses if s != "not_applicable"]
    if not active:
        return "not_applicable"
    if all(s == "passed" for s in active):
        return "passed"
    if any(s == "failed" for s in active):
        return "failed"
    return "review"


def assess_mode_summary(summary: dict, target: PaperTarget, tolerance: StabilityTolerance) -> dict:
    checks = {
        "capacity": _metric_check(float(summary.get("capacity", 0.0)), target.capacity_mbps, tolerance.capacity_percent, True),
        "trigger_rate": _metric_check(float(summary.get("trigger_rate", 0.0)), target.trigger_rate, tolerance.trigger_abs, False),
        "relay_jump": _metric_check(float(summary.get("relay_jump", 0.0)), target.relay_jump_m, tolerance.relay_jump_percent, True),
        "opt_time": _metric_check(float(summary.get("opt_time", 0.0)), target.optimization_time_s, tolerance.optimization_time_percent, True),
        "success_rate": _metric_check(float(summary.get("success_rate", 0.0)), target.success_rate, tolerance.success_abs, False),
    }
    checks["status"] = _combine_status([v["status"] for v in checks.values()])
    return checks


def assess_table3_support(summaries: dict[str, dict], targets: dict[str, PaperTarget] | None = None) -> dict:
    profiles = all_profiles()
    targets = targets or {mode: profile.target for mode, profile in profiles.items()}
    modes = ["full", "no_ppo", "no_trigger", "static"]
    per_mode = {}
    for mode in modes:
        per_mode[mode] = assess_mode_summary(
            summaries.get(mode, {}),
            targets[mode],
            profiles[mode].tolerance,
        )

    capacities = {mode: float(summaries.get(mode, {}).get("capacity", 0.0)) for mode in modes}
    fig14_pass = (
        capacities["full"] > capacities["no_ppo"]
        and capacities["full"] > capacities["no_trigger"]
        and capacities["no_trigger"] > capacities["static"]
    )
    fig14_status = "passed" if fig14_pass else "review"

    jumps = {mode: float(summaries.get(mode, {}).get("relay_jump", 0.0)) for mode in ("full", "no_ppo", "no_trigger")}
    fig13_pass = jumps["full"] <= jumps["no_ppo"] and jumps["full"] <= jumps["no_trigger"]
    fig13_status = "passed" if fig13_pass else "review"

    table_status = _combine_status([per_mode[mode]["status"] for mode in modes])
    overall = _combine_status([fig14_status, fig13_status, table_status])
    return {
        "overall_status": overall,
        "fig14_capacity": {"status": fig14_status, "capacity_means": capacities},
        "fig13_relay_jump": {"status": fig13_status, "relay_jump_means": jumps},
        "table3": {"status": table_status, "per_mode": per_mode},
    }
