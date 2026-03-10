# video_utils.py
"""
Video utilities for recording MountainCar policies to MP4.
"""

import os
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

from fitness_function import PolicyNet, is_continuous_env, unflatten_params


def record_policy_to_mp4(
    genome: np.ndarray,
    hidden_layers: Sequence[int],
    gen: int,
    out_dir: str,
    prefix: str,
    episodes: int,
    max_steps: int,
    video_length_steps: Optional[int] = None,
    record_each_episode: bool = False,
    seed: int = 0,
    env_id: str = "MountainCarContinuous-v0",
) -> Optional[str]:
    """
    Records MP4 with Gymnasium RecordVideo.
    Returns the newest mp4 path if detected, else None.
    """
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    name_prefix = f"{prefix}_{gen:03d}"

    try:
        episode_trigger = (lambda ep: True) if record_each_episode else (lambda ep: ep == 0)
        video_length = int(video_length_steps) if video_length_steps else 0
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=out_dir,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
            video_length=video_length,
            disable_logger=True,
        )
    except Exception as e:
        print(f"[WARN] RecordVideo wrapper failed (gen {gen}): {e}")
        try:
            env.close()
        except Exception:
            pass
        return None

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

    rng = np.random.default_rng(seed + gen * 12345)

    try:
        for _ in range(episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(max_steps):
                action = model.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        env.close()
    except Exception as e:
        print(f"[WARN] Video recording failed (gen {gen}): {e}")
        try:
            env.close()
        except Exception:
            pass
        return None

    try:
        candidates = [
            f for f in os.listdir(out_dir)
            if f.startswith(name_prefix) and f.endswith(".mp4")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(out_dir, fn)))
        return os.path.join(out_dir, candidates[-1])
    except Exception:
        return None
