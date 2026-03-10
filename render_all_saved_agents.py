"""
Render NN images for all saved agent bundles.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Tuple

from PIL import Image

from agent_store import load_agent_bundle
from network_visualizer import save_policy_network_plot


def _iter_agent_paths(base_dir: str, recursive: bool) -> list[str]:
    pattern = os.path.join(base_dir, "**", "*.npz") if recursive else os.path.join(base_dir, "*.npz")
    return sorted(glob.glob(pattern, recursive=recursive), key=_agent_sort_key)


def _agent_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    # Prefer generation-tagged agents first in ascending generation order.
    # Example: progress_best_gen_012.npz
    if "_gen_" in name:
        try:
            gen = int(name.rsplit("_", 1)[-1].split(".")[0])
            return (gen, name)
        except ValueError:
            pass
    # Non-generation bundles (e.g., best_agent.npz) go after generation frames.
    return (10**9, name)


def _extract_generation_from_name(stem: str) -> int | None:
    m = re.search(r"_gen_(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1))


def create_progress_video(
    image_paths: List[str],
    out_video_path: str,
    fps: int = 2,
) -> str:
    """
    Create an MP4 from rendered NN images. Requires ffmpeg binary.
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to create a progression video.")

    out_dir = os.path.dirname(out_video_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nn_frames_") as tmp:
        for i, src in enumerate(image_paths):
            dst = os.path.join(tmp, f"frame_{i:04d}.png")
            shutil.copyfile(src, dst)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmp, "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            out_video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed creating video: {msg}")

    return out_video_path


def create_progress_gif(
    image_paths: List[str],
    out_gif_path: str,
    duration_ms: int = 600,
) -> str:
    """
    Create animated GIF from rendered NN images (no external binaries required).
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to create a progression animation.")

    out_dir = os.path.dirname(out_gif_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    frames = [Image.open(p).convert("RGB") for p in image_paths]
    first, rest = frames[0], frames[1:]
    first.save(
        out_gif_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    for fr in frames:
        fr.close()
    return out_gif_path


def render_all_agents(
    agents_dir: str,
    recursive: bool,
    out_dir: str,
) -> List[str]:
    agent_paths = _iter_agent_paths(agents_dir, recursive)
    if not agent_paths:
        raise FileNotFoundError(f"No .npz agents found in: {agents_dir}")

    print(f"Found {len(agent_paths)} agent bundles.")
    rendered_paths: List[str] = []

    os.makedirs(out_dir, exist_ok=True)
    for agent_path in agent_paths:
        genome, hidden_layers, meta = load_agent_bundle(agent_path)
        env_id = "MountainCarContinuous-v0"
        if isinstance(meta, dict):
            training_cfg = meta.get("training_config")
            if isinstance(training_cfg, dict) and isinstance(training_cfg.get("env_id"), str):
                env_id = training_cfg["env_id"]
        stem = os.path.splitext(os.path.basename(agent_path))[0]
        gen = _extract_generation_from_name(stem)
        out_path = os.path.join(out_dir, f"{stem}_network.png")

        save_policy_network_plot(
            genome=genome,
            hidden_layers=hidden_layers,
            out_path=out_path,
            env_id=env_id,
            generation=gen,
        )
        rendered_paths.append(out_path)
        gen_tag = f"gen={gen:03d}" if gen is not None else "gen=NA"
        print(f"[NN] {gen_tag} {agent_path} -> {out_path}")

    print(f"\nDone. Rendered {len(rendered_paths)} network images.")
    return rendered_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render network images for all saved .npz agents.")
    parser.add_argument(
        "--agents-dir",
        default="agents",
        help="Directory to scan for .npz agent bundles (default: agents).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        default="agents/visualizer",
        help="Output directory for rendered images (default: agents/visualizer).",
    )
    parser.add_argument(
        "--make-video",
        action="store_true",
        help="Create progression video from rendered NN images when possible.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=2,
        help="FPS for progression video (default: 2).",
    )
    args = parser.parse_args()
    rendered_paths = render_all_agents(
        agents_dir=args.agents_dir,
        recursive=args.recursive,
        out_dir=args.out_dir,
    )
    if args.make_video and len(rendered_paths) > 1:
        video_path = os.path.join(args.out_dir, "network_progression.mp4")
        out = create_progress_video(rendered_paths, video_path, fps=args.video_fps)
        print(f"[NN] Progression video: {out}")


if __name__ == "__main__":
    main()
