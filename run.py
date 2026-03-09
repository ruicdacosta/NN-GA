"""
Run training + save + optional best-agent validation recording.
"""

import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Optional

from agent_store import load_agent_bundle, save_agent_bundle
from algorithm import (
    choose_progress_generations,
    prepare_run_dirs,
    run_brkga,
)
from config import BRKGAConfig
from render_all_saved_agents import create_progress_gif, create_progress_video, render_all_agents
from validate_agent import validate_agent
from video_utils import record_policy_to_mp4


def main() -> None:
    cfg = BRKGAConfig()

    run_dir = prepare_run_dirs(cfg)
    print(f"[RUN] {os.path.abspath(run_dir)}")

    prog = []
    if cfg.rec_progress:
        prog = (
            list(range(cfg.generations))
            if cfg.record_every_generation
            else choose_progress_generations(
                generations=cfg.generations,
                include_first=cfg.rec_progress_include_first,
            )
        )
    print(
        f"[REC] rec_progress={cfg.rec_progress} every_gen={cfg.record_every_generation} "
        f"include_first={cfg.rec_progress_include_first} "
        f"gens={prog} (post-training, single-episode, max_steps={cfg.record_max_steps})"
    )

    progress_agent_paths: dict[int, str] = {}
    progress_genomes: dict[int, object] = {}

    def _save_progress_agent(gen: int, genome, fitness: float) -> None:
        progress_genomes[gen] = genome.copy()
        if not cfg.save_progress_agent_bundles:
            return
        if not cfg.save_every_generation_agent_bundle and gen not in prog:
            return
        path = save_agent_bundle(
            genome=genome,
            hidden_layers=cfg.hidden_layers,
            out_dir=cfg.agent_dir,
            prefix=f"{cfg.progress_agent_prefix}_{gen:03d}",
            metadata={
                "seed": cfg.seed,
                "generation": gen,
                "fitness": float(fitness),
                "kind": "progress_best",
            },
        )
        progress_agent_paths[gen] = path
        print(f"   [AGENT] Progress gen {gen:03d}: {path}")

    best_genome, best_fit, _history = run_brkga(
        cfg,
        seed=cfg.seed,
        on_progress_best=_save_progress_agent,
    )
    print(f"\nBest global fitness (avg during eval): {best_fit:.1f}")

    saved_agent_path: Optional[str] = None
    if cfg.save_best_agent_bundle:
        metadata = {
            "seed": cfg.seed,
            "best_fitness": float(best_fit),
            "training_config": asdict(cfg),
        }
        saved_agent_path = save_agent_bundle(
            genome=best_genome,
            hidden_layers=cfg.hidden_layers,
            out_dir=cfg.agent_dir,
            prefix=cfg.agent_prefix,
            metadata=metadata,
        )
        print(f"[AGENT] Saved best agent: {saved_agent_path}")

    if cfg.rec_progress and prog:
        print("\n[REC] Rendering progress videos post-training...")
        for gen in prog:
            try:
                if gen in progress_agent_paths:
                    genome, hidden_layers, _meta = load_agent_bundle(progress_agent_paths[gen])
                elif gen in progress_genomes:
                    genome, hidden_layers = progress_genomes[gen], cfg.hidden_layers
                else:
                    print(f"   [VID] Skip gen {gen:03d}: no checkpoint genome found.")
                    continue

                mp4_path = record_policy_to_mp4(
                    genome=genome,
                    hidden_layers=hidden_layers,
                    gen=gen,
                    out_dir=cfg.video_dir,
                    prefix=cfg.video_prefix,
                    episodes=1,
                    max_steps=cfg.record_max_steps,
                    video_length_steps=cfg.record_max_steps,
                    record_each_episode=False,
                    seed=cfg.seed,
                )
                if mp4_path:
                    print(f"   [VID] Gen {gen:03d}: {mp4_path}")
                else:
                    print(f"   [VID] Gen {gen:03d}: recording failed.")
            except Exception as e:
                print(f"   [VID] Gen {gen:03d}: error: {e}")

    try:
        vis_dir = os.path.join(cfg.agent_dir, "visualizer")
        rendered_paths = render_all_agents(
            agents_dir=cfg.agent_dir,
            recursive=False,
            out_dir=vis_dir,
        )
        print(f"[NN] All agent visualizations: {vis_dir}")
        if len(rendered_paths) > 1:
            gif_path = os.path.join(vis_dir, "network_progression.gif")
            try:
                out_gif = create_progress_gif(rendered_paths, gif_path, duration_ms=600)
                print(f"[NN] Progression animation: {out_gif}")
            except Exception as e:
                print(f"[NN] Progression animation failed: {e}")

            video_path = os.path.join(vis_dir, "network_progression.mp4")
            try:
                out_mp4 = create_progress_video(rendered_paths, video_path, fps=2)
                print(f"[NN] Progression video: {out_mp4}")
            except Exception as e:
                print(f"[NN] Progression MP4 failed (ffmpeg likely missing): {e}")
    except Exception as e:
        print(f"[NN] Batch render failed: {e}")

    if cfg.record_best and saved_agent_path:
        mp4_path = validate_agent(
            agent_path=saved_agent_path,
            out_dir=cfg.video_dir,
            max_steps=cfg.record_max_steps,
            seed=cfg.seed,
            prefix="best_global_validated",
        )
        if mp4_path:
            print(f"[VID] Best validated: {mp4_path}")
        else:
            print("[VID] Best validated: (check videos folder)")

    print(f"\nDone. Videos in: {os.path.abspath(cfg.video_dir)}")
    print(f"Done. Plots in:  {os.path.abspath(cfg.plot_dir)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
