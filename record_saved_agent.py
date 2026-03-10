"""
Record a saved agent bundle to MP4 without retraining.
"""

import os

from agent_store import load_agent_bundle
from video_utils import record_policy_to_mp4


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
    latest_run = find_latest_run_dir("runs")
    agent_path = os.path.join(latest_run, "agents", "best_agent.npz")
    out_dir = os.path.join(latest_run, "videos")
    episodes = 2
    max_steps = 500

    genome, hidden_layers, meta = load_agent_bundle(agent_path)
    env_id = "MountainCarContinuous-v0"
    if isinstance(meta, dict):
        training_cfg = meta.get("training_config")
        if isinstance(training_cfg, dict) and isinstance(training_cfg.get("env_id"), str):
            env_id = training_cfg["env_id"]
    print(f"Loaded agent: {agent_path}")
    if meta:
        print(f"Metadata keys: {sorted(meta.keys())}")

    mp4_path = record_policy_to_mp4(
        genome=genome,
        hidden_layers=hidden_layers,
        gen=0,
        out_dir=out_dir,
        prefix="recorded_from_saved_agent",
        episodes=episodes,
        max_steps=max_steps,
        seed=123,
        env_id=env_id,
    )

    if mp4_path:
        print(f"[VID] {mp4_path}")
    else:
        print("[VID] Recording failed, check logs.")


if __name__ == "__main__":
    main()
