# fitness_function.py
"""
Problem-dependent code for:
- MountainCar-v0 evaluation (Gymnasium)
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
        obs -> hidden... -> logits(actions) -> action = argmax(logits)
    """
    def __init__(
        self,
        obs_dim: int = 4,
        hidden_layers: Sequence[int] = (16,),
        act_dim: int = 2,
        continuous_action: bool = False,
    ):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        self.continuous_action = bool(continuous_action)
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
    def act(self, obs: np.ndarray):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = self.forward(x)
        if self.continuous_action:
            # Continuous MountainCar expects Box action in [-1, 1] with shape (1,)
            a = torch.tanh(out).squeeze(0).cpu().numpy().astype(np.float32)
            return a
        return int(torch.argmax(out, dim=1).item())


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


def get_env_dims(env_id: str = "MountainCarContinuous-v0") -> Tuple[int, int]:
    env = gym.make(env_id)
    obs_dim = int(env.observation_space.shape[0])
    if hasattr(env.action_space, "n"):
        act_dim = int(env.action_space.n)
    else:
        act_dim = int(env.action_space.shape[0])
    if env_id == "MountainCarContinuous-v0" and (obs_dim != 2 or act_dim != 1):
        env.close()
        raise ValueError("MountainCarContinuous-v0 spec mismatch: expected obs_dim=2 and act_dim=1.")
    env.close()
    return obs_dim, act_dim


def is_continuous_env(env_id: str) -> bool:
    return env_id == "MountainCarContinuous-v0"


def make_template_model(
    hidden_layers: Sequence[int],
    env_id: str = "MountainCarContinuous-v0",
) -> PolicyNet:
    obs_dim, act_dim = get_env_dims(env_id)
    return PolicyNet(
        obs_dim=obs_dim,
        hidden_layers=hidden_layers,
        act_dim=act_dim,
        continuous_action=is_continuous_env(env_id),
    )


def rollout_fitness(
    genome: np.ndarray,
    hidden_layers: Sequence[int],
    episodes: int,
    max_steps: int,
    seed: int,
    env_id: str = "MountainCarContinuous-v0",
    use_shaped_fitness: bool = True,
    mountaincar_progress_scale: float = 1000.0,
    mountaincar_velocity_scale: float = 50.0,
    mountaincar_goal_bonus: float = 1000.0,
) -> float:
    """
    Fitness = average episodic score over `episodes`.
    For MountainCar-v0, shaped fitness is used by default to avoid flat rewards.
    """
    env = gym.make(env_id)
    obs_dim = int(env.observation_space.shape[0])
    if hasattr(env.action_space, "n"):
        act_dim = int(env.action_space.n)
    else:
        act_dim = int(env.action_space.shape[0])

    model = PolicyNet(
        obs_dim=obs_dim,
        hidden_layers=hidden_layers,
        act_dim=act_dim,
        continuous_action=is_continuous_env(env_id),
    )
    unflatten_params(model, genome)

    rng = np.random.default_rng(seed)
    total = 0.0

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_reward = 0.0
        max_pos = float(obs[0])
        vel_abs_sum = 0.0
        reached_goal = False
        steps = 0

        for step in range(max_steps):
            action = model.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            max_pos = max(max_pos, float(obs[0]))
            vel_abs_sum += abs(float(obs[1]))
            steps = step + 1
            if terminated or truncated:
                reached_goal = bool(terminated)
                break

        if env_id.startswith("MountainCar") and use_shaped_fitness:
            # Dense signal:
            # - main driver: furthest position reached in episode
            # - tie-breaker: velocity magnitude (encourages momentum building)
            # - strong bonus for actually reaching the goal, with faster completion preferred
            goal_pos = float(getattr(env.unwrapped, "goal_position", 0.45))
            progress_norm = np.clip((max_pos - (-1.2)) / (goal_pos - (-1.2)), 0.0, 1.0)
            score = progress_norm * mountaincar_progress_scale
            score += vel_abs_sum * mountaincar_velocity_scale / float(max(1, steps))
            if reached_goal:
                score += mountaincar_goal_bonus + float(max_steps - steps)
            total += score
        else:
            total += ep_reward

    env.close()
    return total / float(episodes)
