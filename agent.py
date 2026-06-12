from __future__ import annotations

import json
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None
    F = None
    TORCH_AVAILABLE = False


class LightweightPPOAgent:
    """Lightweight PPO tuner for RLPSOEC PSO hyperparameters.

    If PyTorch is unavailable, the public API remains usable through a
    deterministic fallback tuner. That keeps simulation and regression scripts
    runnable in lightweight environments while preserving the PPO path where
    the original dependency is installed.
    """

    def __init__(
        self,
        state_dim=4,
        action_dim=3,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=3,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.torch_available = TORCH_AVAILABLE
        self.update_count = 0
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        if not self.torch_available:
            self.rng = np.random.RandomState(2026)
            print("[PPO] Torch not found; using deterministic RLPSOEC tuner fallback")
            return

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh(),
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    def extract_state(self, env_info):
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

    def get_action(self, state, deterministic=False):
        if not self.torch_available:
            state = np.asarray(state, dtype=np.float32)
            action = np.array(
                [
                    0.15 * state[0] + 0.10 * state[3],
                    -0.10 + 0.12 * state[2],
                    0.05 - 0.10 * state[0],
                ],
                dtype=np.float32,
            )
            if not deterministic:
                action += self.rng.normal(0.0, self.entropy_coef, size=3).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            return action, 0.0, 0.0

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_tensor)
        if deterministic:
            action = logits
        else:
            noise = torch.randn_like(logits) * self.entropy_coef
            action = logits + noise
        action = action.clamp(-1, 1)
        log_prob = -0.5 * ((action - logits) ** 2).sum(dim=1)
        value = self.critic(state_tensor).squeeze(1)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def map_action_to_params(self, action):
        a0, a1, a2 = np.asarray(action, dtype=float)
        return {
            "search_radius_multiplier": float(0.5 + (a0 + 1.0) / 2.0),
            "population_multiplier": float(0.5 + (a1 + 1.0) / 2.0),
            "update_frequency": int(1 + ((a2 + 1.0) / 2.0) * 9.0),
        }

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages(self):
        if not self.torch_available:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        returns = []
        running_return = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                running_return = 0
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        values = torch.tensor(self.values, device=self.device, dtype=torch.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        if not self.torch_available:
            self._clear_buffer()
            self.update_count += 1
            return

        if len(self.states) < 32:
            return

        states = torch.tensor(self.states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(self.log_probs, device=self.device, dtype=torch.float32)
        returns, advantages = self.compute_advantages()

        for _ in range(self.k_epochs):
            logits = self.actor(states)
            log_probs = -0.5 * ((actions - logits) ** 2).sum(dim=1)
            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            values = self.critic(states).squeeze(1)
            value_loss = F.mse_loss(values, returns)
            entropy = -(logits**2).sum(dim=1).mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

        self._clear_buffer()
        self.update_count += 1
        print(f"[PPO Update #{self.update_count}]")

    def compute_reward(self, comm_improvement, convergence_speed):
        reward = comm_improvement * 10.0 + convergence_speed * 2.0
        return float(np.clip(reward, -10, 10))

    def save_model(self, path):
        if not self.torch_available:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"fallback": "deterministic_rlpsoec_tuner", "update_count": self.update_count},
                    f,
                    indent=2,
                )
            print(f"[PPO] fallback tuner state saved to {path}")
            return

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
            },
            path,
        )
        print(f"[PPO] model saved to {path}")

    def load_model(self, path):
        if not self.torch_available:
            if os.path.exists(path):
                print(f"[PPO] fallback mode ignores Torch checkpoint {path}")
            else:
                print(f"[PPO] no checkpoint found at {path}; using fallback tuner")
            return

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.update_count = ckpt.get("update_count", 0)
            print(f"[PPO] loaded model {path}")
        else:
            print(f"[PPO] no checkpoint found at {path}; using random initialization")

    def _clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
