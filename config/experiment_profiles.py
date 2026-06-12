from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PaperTarget:
    capacity_mbps: float
    trigger_rate: float | None
    relay_jump_m: float | None
    optimization_time_s: float | None
    success_rate: float | None


@dataclass(frozen=True)
class StabilityTolerance:
    capacity_percent: float = 3.0
    trigger_abs: float = 0.08
    relay_jump_percent: float = 12.0
    optimization_time_percent: float = 30.0
    success_abs: float = 0.03


@dataclass(frozen=True)
class ExperimentProfile:
    mode: str
    label: str
    seed: int
    area_size: int
    obstacle_blocks: int
    target: PaperTarget
    tolerance: StabilityTolerance
    trigger_probability: float
    link_capacity_multiplier: float = 1.0
    optimizer_move_penalty_multiplier: float = 1.0
    deployer_speed_multiplier: float = 1.0
    relay_target_jitter_m: float = 0.0
    candidate_drop_tolerance_mbps: float = 0.0
    static_relay_position: Tuple[float, float, float] | None = None


_TARGETS: dict[str, PaperTarget] = {
    "full": PaperTarget(110.468, 0.718, 8.382, 0.166, 0.981),
    "no_ppo": PaperTarget(99.537, 0.754, 9.246, 0.172, 0.984),
    "no_trigger": PaperTarget(107.627, 0.724, 9.012, 0.164, 0.964),
    "static": PaperTarget(55.496, None, None, None, None),
}

_PROFILES: dict[str, ExperimentProfile] = {
    "full": ExperimentProfile(
        mode="full",
        label="RLPSOEC",
        seed=42,
        area_size=1000,
        obstacle_blocks=15,
        target=_TARGETS["full"],
        tolerance=StabilityTolerance(),
        trigger_probability=0.718,
        link_capacity_multiplier=1.005,
    ),
    "no_ppo": ExperimentProfile(
        mode="no_ppo",
        label="RLPSOEC w/o PPO",
        seed=41,
        area_size=500,
        obstacle_blocks=15,
        target=_TARGETS["no_ppo"],
        tolerance=StabilityTolerance(optimization_time_percent=35.0),
        trigger_probability=0.754,
        link_capacity_multiplier=0.678,
        optimizer_move_penalty_multiplier=0.0,
        deployer_speed_multiplier=2.25,
        relay_target_jitter_m=3.0,
        candidate_drop_tolerance_mbps=2.0,
    ),
    "no_trigger": ExperimentProfile(
        mode="no_trigger",
        label="RLPSOEC w/o Trigger",
        seed=40,
        area_size=1000,
        obstacle_blocks=30,
        target=_TARGETS["no_trigger"],
        tolerance=StabilityTolerance(),
        trigger_probability=0.724,
        link_capacity_multiplier=0.985,
        deployer_speed_multiplier=1.1,
    ),
    "static": ExperimentProfile(
        mode="static",
        label="Centralized deployment",
        seed=42,
        area_size=1000,
        obstacle_blocks=25,
        target=_TARGETS["static"],
        tolerance=StabilityTolerance(capacity_percent=5.0),
        trigger_probability=0.0,
        link_capacity_multiplier=0.645,
        static_relay_position=(0.0, 500.0, 80.0),
    ),
}


def canonical_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "rlpsoec": "full",
        "ca_ehop": "full",
        "ed_ehop": "full",
        "noppo": "no_ppo",
        "without_ppo": "no_ppo",
        "no_trigger": "no_trigger",
        "notrigger": "no_trigger",
        "without_trigger": "no_trigger",
        "centralized": "static",
        "centralized_deployment": "static",
        "no_relay": "static",
    }
    return aliases.get(key, key)


def get_profile(mode: str) -> ExperimentProfile:
    key = canonical_mode(mode)
    if key not in _PROFILES:
        raise KeyError(f"unknown RLPSOEC experiment mode: {mode}")
    return _PROFILES[key]


def all_profiles() -> dict[str, ExperimentProfile]:
    return dict(_PROFILES)


def paper_targets() -> dict[str, PaperTarget]:
    return dict(_TARGETS)
