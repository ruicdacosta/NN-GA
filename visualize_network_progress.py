"""
Render network visualizations across saved progress generations.
"""

import argparse
import glob
import os
import re

from agent_store import load_agent_bundle
from network_visualizer import find_latest_run_dir, save_policy_network_plot


def _gen_from_path(path: str) -> int:
    m = re.search(r"_(\d{3})\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Render network progression across generations.")
    parser.add_argument("--run-root", default="runs", help="Root directory containing run folders.")
    parser.add_argument("--run-dir", default=None, help="Specific run dir. If omitted, latest run is used.")
    parser.add_argument("--agent-prefix", default="progress_best_gen", help="Prefix used for progress agent bundles.")
    parser.add_argument("--out-dir", default=None, help="Output folder for generated png frames.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(args.run_root)
    agents_dir = os.path.join(run_dir, "agents")
    plots_dir = os.path.join(run_dir, "plots")
    out_dir = args.out_dir or os.path.join(plots_dir, "network_progress")
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(agents_dir, f"{args.agent_prefix}_*.npz")
    agent_paths = sorted(glob.glob(pattern), key=_gen_from_path)

    if not agent_paths:
        raise FileNotFoundError(f"No progress agent bundles found with pattern: {pattern}")

    print(f"[RUN] {run_dir}")
    print(f"[OUT] {out_dir}")

    for p in agent_paths:
        gen = _gen_from_path(p)
        genome, hidden_layers, meta = load_agent_bundle(p)
        env_id = "MountainCarContinuous-v0"
        if isinstance(meta, dict):
            training_cfg = meta.get("training_config")
            if isinstance(training_cfg, dict) and isinstance(training_cfg.get("env_id"), str):
                env_id = training_cfg["env_id"]
        out_path = os.path.join(out_dir, f"network_gen_{gen:03d}.png")
        save_policy_network_plot(
            genome=genome,
            hidden_layers=hidden_layers,
            out_path=out_path,
            env_id=env_id,
        )
        print(f"[NN] Gen {gen:03d}: {out_path}")

    print("\nDone. You can compare frames in order or build a video externally if needed.")


if __name__ == "__main__":
    main()
