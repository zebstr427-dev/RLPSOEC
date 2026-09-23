from dataclasses import dataclass
from typing import List

import numpy as np

from core.config import (
    PPO_ACTION_DIM,
    PPO_CLIP_EPSILON,
    PPO_ENTROPY_COEFFICIENT,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_MAX_GRAD_NORM,
    PPO_MIN_BUFFER,
    PPO_STATE_DIM,
    PPO_VALUE_COEFFICIENT,
)

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:
    torch = None
    nn = None
    Normal = None


if nn is not None:
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(PPO_STATE_DIM, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.mean = nn.Linear(64, PPO_ACTION_DIM)
            self.value = nn.Linear(64, 1)
            self.log_std = nn.Parameter(torch.zeros(PPO_ACTION_DIM))

        def forward(self, state):
            hidden = self.body(state)
            return self.mean(hidden), self.log_std.expand_as(self.mean(hidden)), self.value(hidden).squeeze(-1)
else:
    ActorCritic = None


@dataclass
class Transition:
    state: np.ndarray
    raw_action: np.ndarray
    log_probability: float
    value: float
    reward: float


class PPOController:
    def __init__(self, seed: int):
        if torch is None:
            raise RuntimeError("torch is required for PPO")
        torch.manual_seed(int(seed))
        self.network = ActorCritic()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=PPO_LEARNING_RATE)
        self.transitions: List[Transition] = []
        self.update_count = 0

    def act(self, state):
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            mean, log_std, value = self.network(state_tensor)
            distribution = Normal(mean, log_std.exp())
            raw = distribution.sample()
            action = torch.tanh(raw)
            log_probability = distribution.log_prob(raw).sum(-1) - torch.log(
                1.0 - action.pow(2) + 1e-6
            ).sum(-1)
        return (
            action.squeeze(0).cpu().numpy(),
            raw.squeeze(0).cpu().numpy(),
            float(log_probability.item()),
            float(value.item()),
        )

    @staticmethod
    def map_action(action):
        bounded = np.asarray(action, dtype=float)
        return {
            "search_scale": float(0.5 + 0.75 * (bounded[0] + 1.0)),
            "population": int(np.clip(np.rint(10.0 + 45.0 * (bounded[1] + 1.0)), 10, 100)),
            "cadence": int(np.clip(np.rint(1.0 + 4.5 * (bounded[2] + 1.0)), 1, 10)),
        }

    def store(self, state, raw_action, log_probability, value, reward):
        self.transitions.append(Transition(
            state=np.asarray(state, dtype=np.float32),
            raw_action=np.asarray(raw_action, dtype=np.float32),
            log_probability=float(log_probability),
            value=float(value),
            reward=float(reward),
        ))

    def update(self):
        if len(self.transitions) < PPO_MIN_BUFFER:
            return False
        states = torch.as_tensor(np.asarray([item.state for item in self.transitions]), dtype=torch.float32)
        raw_actions = torch.as_tensor(np.asarray([item.raw_action for item in self.transitions]), dtype=torch.float32)
        old_log_probabilities = torch.as_tensor(
            [item.log_probability for item in self.transitions], dtype=torch.float32
        )
        values = torch.as_tensor([item.value for item in self.transitions], dtype=torch.float32)
        rewards = [item.reward for item in self.transitions]
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = float(reward) + PPO_GAMMA * running
            returns.append(running)
        returns = torch.as_tensor(list(reversed(returns)), dtype=torch.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(PPO_EPOCHS):
            mean, log_std, predicted_values = self.network(states)
            distribution = Normal(mean, log_std.exp())
            actions = torch.tanh(raw_actions)
            log_probabilities = distribution.log_prob(raw_actions).sum(-1) - torch.log(
                1.0 - actions.pow(2) + 1e-6
            ).sum(-1)
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (returns - predicted_values).pow(2).mean()
            entropy = distribution.entropy().sum(-1).mean()
            loss = policy_loss + PPO_VALUE_COEFFICIENT * value_loss - PPO_ENTROPY_COEFFICIENT * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), PPO_MAX_GRAD_NORM)
            self.optimizer.step()
        self.transitions.clear()
        self.update_count += 1
        return True
