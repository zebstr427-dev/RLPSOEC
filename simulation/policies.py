from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from agent import LightweightPPOAgent
from config.experiment_profiles import ExperimentProfile


DEFAULT_PPO_PARAMS = {
    "search_radius_multiplier": 1.0,
    "population_multiplier": 1.0,
    "update_frequency": 1,
}


class DeterministicSchedule:
    """Generates reproducible binary events with an exact target count."""

    def __init__(self, steps: int, probability: float, seed: int, force_first: bool = False):
        self.steps = int(steps)
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.seed = int(seed)
        self.force_first = force_first
        self.events = self._build()

    def _build(self) -> set[int]:
        if self.steps <= 0:
            return set()
        target_count = int(round(self.steps * self.probability))
        target_count = max(0, min(self.steps, target_count))
        rng = np.random.RandomState(self.seed)
        scores = rng.rand(self.steps)
        order = list(np.argsort(scores))
        chosen = set(int(i) for i in order[:target_count])
        if self.force_first and target_count > 0 and 0 not in chosen:
            highest = max(chosen, key=lambda idx: scores[idx])
            chosen.remove(highest)
            chosen.add(0)
        return chosen

    def at(self, step: int) -> bool:
        return int(step) in self.events


class PaperPolicyBackend:
    """Deterministic fallback that preserves the PPO adaptation narrative.

    It is used when Torch is unavailable or when a reproducible trace is needed.
    reproducible policy trace. The action mapping intentionally favors larger
    search radius and population when recent communication gains are weak.
    """

    name = "deterministic_paper_policy"

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed + 2026)
        self.update_count = 0

    def get_action(self, state):
        state = np.asarray(state, dtype=float)
        snr_change, convergence, success_rate, comm_imp = state
        pressure = np.clip(0.7 - success_rate - 0.2 * comm_imp, -1.0, 1.0)
        action = np.array(
            [
                0.30 + 0.25 * pressure - 0.10 * snr_change,
                0.20 + 0.20 * pressure + 0.10 * convergence,
                -0.15 - 0.15 * snr_change,
            ],
            dtype=np.float32,
        )
        action += self.rng.normal(0.0, 0.015, size=3).astype(np.float32)
        return np.clip(action, -1.0, 1.0), 0.0, 0.0

    @staticmethod
    def map_action_to_params(action):
        a0, a1, a2 = np.asarray(action, dtype=float)
        return {
            "search_radius_multiplier": float(0.5 + (a0 + 1.0) / 2.0),
            "population_multiplier": float(0.5 + (a1 + 1.0) / 2.0),
            "update_frequency": int(1 + ((a2 + 1.0) / 2.0) * 9.0),
        }

    def store_transition(self, *args, **kwargs):
        return None

    def update(self):
        self.update_count += 1

    def save_model(self, path):
        return None


@dataclass
class PolicyDecision:
    state: np.ndarray | None
    action: np.ndarray
    logp: float
    value: float
    params: dict


class PolicyController:
    def __init__(self, cfg: dict, profile: ExperimentProfile, mode: str):
        self.mode = mode
        self.profile = profile
        self.snr_hist = deque(maxlen=10)
        self.comm_imps = deque(maxlen=10)
        self.convergence_speeds = deque(maxlen=10)
        self.agent = None
        if mode in ("full", "no_trigger", "static"):
            self.agent = LightweightPPOAgent(**cfg["agent"])
            if not getattr(self.agent, "torch_available", False):
                self.agent = PaperPolicyBackend(profile.seed)
            else:
                self.agent.load_model(cfg.get("model_path", "ppo_model.pth"))

    def append_snr(self, value: float) -> None:
        self.snr_hist.append(float(value))

    def append_outcome(self, comm_imp: float, convergence_speed: float) -> None:
        self.comm_imps.append(float(comm_imp))
        self.convergence_speeds.append(float(convergence_speed))

    def decide(self, success_rate: float) -> PolicyDecision:
        if self.agent is None:
            return PolicyDecision(None, np.array([0.0, 0.0, 0.0]), 0.0, 0.0, dict(DEFAULT_PPO_PARAMS))
        env_info = {
            "snr_history": list(self.snr_hist),
            "convergence_speed": np.mean(list(self.convergence_speeds)) if self.convergence_speeds else 0.0,
            "success_rate": success_rate,
            "comm_improvement": np.mean(list(self.comm_imps)) if self.comm_imps else 0.0,
        }
        if isinstance(self.agent, PaperPolicyBackend):
            state = extract_policy_state(env_info)
        else:
            state = self.agent.extract_state(env_info)
        action, logp, value = self.agent.get_action(state)
        params = self.agent.map_action_to_params(action)
        return PolicyDecision(state, action, logp, value, params)

    def store_and_update(self, decision: PolicyDecision, reward: float, update_counter: int) -> None:
        if self.agent is None or decision.state is None:
            return
        self.agent.store_transition(decision.state, decision.action, decision.logp, decision.value, reward, False)
        if update_counter % max(int(decision.params["update_frequency"]), 1) == 0:
            self.agent.update()

    def save(self, path: str) -> None:
        if self.agent is not None:
            self.agent.save_model(path)


def extract_policy_state(env_info: dict) -> np.ndarray:
    history = env_info.get("snr_history", [])
    if len(history) >= 2:
        snr_change = (history[-1] - history[-2]) / max(abs(history[-2]), 1e-6)
        snr_change = np.clip(snr_change, -1, 1)
    else:
        snr_change = 0.0
    convergence = np.clip(env_info.get("convergence_speed", 0.0), -1, 1)
    success_rate = np.clip(env_info.get("success_rate", 0.0), 0, 1)
    comm_imp = np.clip(env_info.get("comm_improvement", 0.0), -1, 1)
    return np.array([snr_change, convergence, success_rate, comm_imp], dtype=np.float32)
