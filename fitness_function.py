# fitness_function.py
"""
Problem-dependent code for:
- CartPole-v1 evaluation (Gymnasium)
- PyTorch policy network
- genome <-> model parameter conversion
- rollout fitness
"""

from typing import Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    MLP:
        obs(4) -> hidden... -> logits(2) -> action = argmax(logits)
    """
    def __init__(
        self,
        obs_dim: int = 4,
        hidden_layers: Sequence[int] = (16,),
        act_dim: int = 2,
    ):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        dims = [obs_dim, *list(hidden_layers), act_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        return int(torch.argmax(logits, dim=1).item())


def flatten_params(model: nn.Module) -> np.ndarray:
    """Genome: 1D vector with all parameters."""
    parts = []
    for p in model.parameters():
        parts.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(parts).astype(np.float64)


def unflatten_params(model: nn.Module, genome: np.ndarray) -> None:
    """Load 1D genome into PyTorch model parameters."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = genome[offset:offset + n]
            offset += n
            p.copy_(torch.tensor(chunk.reshape(p.shape), dtype=p.dtype))
    if offset != len(genome):
        raise ValueError(
            f"Genome has incorrect length: used {offset}, got {len(genome)}"
        )


def get_cartpole_dims() -> Tuple[int, int]:
    env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)
    env.close()
    return obs_dim, act_dim


def make_template_model(hidden_layers: Sequence[int]) -> PolicyNet:
    obs_dim, act_dim = get_cartpole_dims()
    return PolicyNet(obs_dim=obs_dim, hidden_layers=hidden_layers, act_dim=act_dim)


def rollout_fitness(
    genome: np.ndarray,
    hidden_layers: Sequence[int],
    episodes: int,
    max_steps: int,
    seed: int,
) -> float:
    """
    Fitness = average episodic reward over `episodes`.
    """
    env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    model = PolicyNet(obs_dim=obs_dim, hidden_layers=hidden_layers, act_dim=act_dim)
    unflatten_params(model, genome)

    rng = np.random.default_rng(seed)
    total = 0.0

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_reward = 0.0

        for _ in range(max_steps):
            action = model.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                break

        total += ep_reward

    env.close()
    return total / float(episodes)
