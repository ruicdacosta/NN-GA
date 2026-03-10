"""Validate a saved agent by recording one full rollout episode."""

import os
from typing import Optional

from agent_store import load_agent_bundle
from video_utils import record_policy_to_mp4


def validate_agent(
    agent_path: str,
    out_dir: str,
    max_steps: int = 999,
    seed: int = 123,
    prefix: str = "validated_agent",
) -> Optional[str]:
    """
    Record one full episode from reset to termination/truncation.
    """
    genome, hidden_layers, meta = load_agent_bundle(agent_path)
    env_id = "MountainCarContinuous-v0"
    if isinstance(meta, dict):
        training_cfg = meta.get("training_config")
        if isinstance(training_cfg, dict) and isinstance(training_cfg.get("env_id"), str):
            env_id = training_cfg["env_id"]

    return record_policy_to_mp4(
        genome=genome,
        hidden_layers=hidden_layers,
        gen=0,
        out_dir=out_dir,
        prefix=prefix,
        episodes=1,
        max_steps=max_steps,
        video_length_steps=max_steps,
        record_each_episode=False,
        seed=seed,
        env_id=env_id,
    )


def find_latest_run_dir(run_root: str = "runs") -> str:
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"Run root not found: {run_root}")
    candidates = [
        os.path.join(run_root, d)
        for d in os.listdir(run_root)
        if os.path.isdir(os.path.join(run_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run folders found in: {run_root}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main() -> None:
    run_dir = find_latest_run_dir("runs")
    path = os.path.join(run_dir, "agents", "best_agent.npz")
    video_dir = os.path.join(run_dir, "videos")
    video = validate_agent(agent_path=path, out_dir=video_dir)
    if video:
        print(f"[VID] {video}")
    else:
        print("[VID] Validation recording failed.")


if __name__ == "__main__":
    main()
